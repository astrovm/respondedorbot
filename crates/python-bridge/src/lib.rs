//! Temporary Python bridge for incrementally adopting `bot-core`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use bot_adapters::compaction_job::normalize_compaction_job as normalize_compaction_job_adapter;
use bot_adapters::redis_chat_admin::{
    cache_chat_admin as cache_chat_admin_adapter,
    chat_admin_cache_key as chat_admin_cache_key_adapter,
    get_cached_chat_admin as get_cached_chat_admin_adapter,
};
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_media_cache::{
    cache_media as cache_media_adapter, get_cached_media as get_cached_media_adapter,
    media_cache_key as media_cache_key_adapter,
};
use bot_core::admin_reports::{
    CreditLogLimit, parse_creditlog_limit as parse_creditlog_limit_core,
    truncate_report as truncate_admin_report_core,
};
use bot_core::base_conversion::{BaseConversion, convert_base as convert_base_core};
use bot_core::cache_policy::{
    CacheDecision, evaluate_cache as evaluate_cache_core,
    last_success_ttl as last_success_ttl_core, request_cache_history_key as cache_history_key_core,
    request_cache_key as cache_key_core, request_cache_ttl as cache_ttl_core,
};
use bot_core::command_normalization::normalize_command_text as normalize_command_text_core;
use bot_core::command_parsing::parse_command as parse_command_core;
use bot_core::compaction_policy::{
    CompactionDisposition, evaluate_compaction as evaluate_compaction_core,
    is_due as compaction_is_due_core, retry_after_failure as retry_compaction_core,
};
use bot_core::config_callbacks::{
    ConfigCallbackEvaluation, ConfigCallbackInput, ToggleField,
    evaluate_config_callback as evaluate_config_callback_core,
};
use bot_core::credit_units::{
    CreditUnits, format_credit_units as format_credit_units_core,
    parse_credit_units as parse_credit_units_core,
    rescale_credit_units as rescale_credit_units_core,
    whole_credits_to_units as whole_credits_to_units_core,
};
use bot_core::devo::{
    DevoInput, DevoQuotes, DevoResult, calculate_devo as calculate_devo_core,
    parse_devo_input as parse_devo_input_core,
};
use bot_core::hacker_news::{
    HackerNewsRenderItem, format_items as format_hacker_news_core,
    normalize_feed_item as normalize_hacker_news_item_core,
};
use bot_core::links::{
    select_unique_urls as select_unique_urls_core, trim_detected_url as trim_detected_url_core,
    utf16_slice as utf16_slice_core,
};
use bot_core::market_context::{
    CryptoQuote as MarketCryptoQuote, DollarQuote as MarketDollarQuote, MarketSnapshot,
    format_market_context as format_market_context_core,
};
use bot_core::market_models::{
    MarketModel, Valuation, evaluate_market_model as evaluate_market_model_core,
};
use bot_core::message_state::{
    SearchCandidate, bot_message_metadata_key as bot_metadata_key_core,
    chat_compacted_until_key as compacted_key_core, chat_members_key as members_key_core,
    chat_summary_key as summary_key_core, escape_search_tag as escape_search_tag_core,
    escape_search_text as escape_search_text_core,
    prepare_chat_member_payload as prepare_chat_member_core,
    prepare_message_write as prepare_message_write_core,
    rank_search_candidates as rank_search_candidates_core,
    user_chat_compacted_until_key as user_compacted_key_core,
    user_chat_summary_key as user_summary_key_core,
};
use bot_core::polymarket::{MarketOutcome, rank_outcomes as rank_outcomes_core};
use bot_core::price_queries::{
    AmountConversion, PriceQuery, ProviderScope, parse_price_query as parse_price_query_core,
};
use bot_core::random_reply::{
    RandomAnswer, RandomSuffix, evaluate_random_reply as evaluate_random_reply_core,
};
use bot_core::random_selection::{
    RandomSelection, parse_random_selection as parse_random_selection_core,
};
use bot_core::routing::{
    MediaRoutingInput, ResponseRoutingEvaluation, ResponseRoutingInput,
    evaluate_response_routing as evaluate_response_routing_core,
    should_auto_process_media as should_auto_process_media_core,
};
use bot_core::rulo::{
    ExchangeQuote, RuloDetail, RuloEvaluation, RuloInput, evaluate_rulo as evaluate_rulo_core,
};
use bot_core::satoshi::format_satoshi_quote as format_satoshi_quote_core;
use bot_core::task_triggers::{
    IntegerInput, TaskTrigger, TaskTriggerInput, TriggerConfigInput, TriggerError,
    parse_task_trigger as parse_task_trigger_core,
};
use bot_core::weather::{
    select_forecast_hour as select_forecast_hour_core,
    select_location_candidate as select_location_candidate_core,
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

#[derive(Serialize)]
struct MarketModelDto {
    value: String,
    percentage: String,
    valuation: &'static str,
}

#[derive(Serialize)]
struct DevoResultDto {
    profit: String,
    fee: String,
    official: String,
    usdt: String,
    card: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    purchase: Option<DevoPurchaseDto>,
}

#[derive(Serialize)]
struct DevoPurchaseDto {
    usd: String,
    ars: String,
    usdt: String,
    profit_ars: String,
    profit_usdt: String,
    total_ars: String,
    total_usdt: String,
}

#[derive(Deserialize)]
struct RuloInputDto {
    official: Option<f64>,
    mep: Option<f64>,
    blue: Option<f64>,
    usd_to_usdt: Vec<ExchangeQuoteDto>,
    usdt_to_ars: Vec<ExchangeQuoteDto>,
    usd_amount: f64,
}

#[derive(Deserialize)]
struct MarketOutcomeInputDto {
    title: String,
    cached_probability: f64,
    live_probability: Option<f64>,
}

#[derive(Serialize)]
struct RankedOutcomeDto {
    title: String,
    percentage: f64,
}

#[derive(Serialize)]
struct HackerNewsItemDto {
    title: String,
    url: String,
    points: Option<i64>,
    comments: Option<i64>,
    comments_url: String,
}

#[derive(Deserialize)]
struct HackerNewsRenderItemDto {
    title: String,
    url: String,
    points: Option<i64>,
    comments: Option<i64>,
    comments_url: String,
}

#[derive(Deserialize)]
struct ConfigCallbackInputDto {
    action: String,
    value: String,
    current_toggle: Option<bool>,
    current_creditless_limit: Option<i64>,
    numeric_value: Option<i64>,
    timezone_min: i64,
    timezone_max: i64,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ConfigCallbackEvaluationDto {
    NoChange,
    GuardCurrent,
    InvalidTimezone,
    InvalidCreditlessLimit,
    SetLanguage { value: String },
    SetLinkMode { value: String },
    SetToggle { field: &'static str, value: bool },
    SetTimezone { value: i64 },
    SetCreditlessLimit { value: i64 },
}

impl From<ConfigCallbackEvaluation> for ConfigCallbackEvaluationDto {
    fn from(value: ConfigCallbackEvaluation) -> Self {
        match value {
            ConfigCallbackEvaluation::NoChange => Self::NoChange,
            ConfigCallbackEvaluation::GuardCurrent => Self::GuardCurrent,
            ConfigCallbackEvaluation::InvalidTimezone => Self::InvalidTimezone,
            ConfigCallbackEvaluation::InvalidCreditlessLimit => Self::InvalidCreditlessLimit,
            ConfigCallbackEvaluation::SetLanguage(value) => Self::SetLanguage { value },
            ConfigCallbackEvaluation::SetLinkMode(value) => Self::SetLinkMode { value },
            ConfigCallbackEvaluation::SetToggle { field, value } => Self::SetToggle {
                field: match field {
                    ToggleField::RandomReplies => "ai_random_replies",
                    ToggleField::CommandFollowups => "ai_command_followups",
                    ToggleField::IgnoreLinkFixFollowups => "ignore_link_fix_followups",
                },
                value,
            },
            ConfigCallbackEvaluation::SetTimezone(value) => Self::SetTimezone { value },
            ConfigCallbackEvaluation::SetCreditlessLimit(value) => {
                Self::SetCreditlessLimit { value }
            }
        }
    }
}

#[derive(Deserialize)]
struct ExchangeQuoteDto {
    exchange: String,
    price: Option<f64>,
}

impl From<RuloInputDto> for RuloInput {
    fn from(value: RuloInputDto) -> Self {
        let convert_quotes = |quotes: Vec<ExchangeQuoteDto>| {
            quotes
                .into_iter()
                .map(|quote| ExchangeQuote {
                    exchange: quote.exchange,
                    price: quote.price,
                })
                .collect()
        };
        Self {
            official: value.official,
            mep: value.mep,
            blue: value.blue,
            usd_to_usdt: convert_quotes(value.usd_to_usdt),
            usdt_to_ars: convert_quotes(value.usdt_to_ars),
            usd_amount: value.usd_amount,
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum RuloEvaluationDto {
    OfficialError,
    Routes {
        official: String,
        base_usd: String,
        base_ars: String,
        routes: Vec<RuloRouteDto>,
    },
}

#[derive(Serialize)]
struct RuloRouteDto {
    label: &'static str,
    sell_price: String,
    difference: String,
    percentage: String,
    details: Vec<RuloDetailDto>,
}

#[derive(Serialize)]
#[serde(tag = "kind", content = "text", rename_all = "snake_case")]
enum RuloDetailDto {
    Steps(String),
    Result(String),
    Profit(String),
}

impl From<RuloEvaluation> for RuloEvaluationDto {
    fn from(value: RuloEvaluation) -> Self {
        match value {
            RuloEvaluation::OfficialError => Self::OfficialError,
            RuloEvaluation::Routes(plan) => Self::Routes {
                official: plan.official,
                base_usd: plan.base_usd,
                base_ars: plan.base_ars,
                routes: plan
                    .routes
                    .into_iter()
                    .map(|route| RuloRouteDto {
                        label: route.label,
                        sell_price: route.sell_price,
                        difference: route.difference,
                        percentage: route.percentage,
                        details: route
                            .details
                            .into_iter()
                            .map(|detail| match detail {
                                RuloDetail::Steps(text) => RuloDetailDto::Steps(text),
                                RuloDetail::Result(text) => RuloDetailDto::Result(text),
                                RuloDetail::Profit(text) => RuloDetailDto::Profit(text),
                            })
                            .collect(),
                    })
                    .collect(),
            },
        }
    }
}

impl From<DevoResult> for DevoResultDto {
    fn from(value: DevoResult) -> Self {
        Self {
            profit: value.profit,
            fee: value.fee,
            official: value.official,
            usdt: value.usdt,
            card: value.card,
            purchase: value.purchase.map(|purchase| DevoPurchaseDto {
                usd: purchase.usd,
                ars: purchase.ars,
                usdt: purchase.usdt,
                profit_ars: purchase.profit_ars,
                profit_usdt: purchase.profit_usdt,
                total_ars: purchase.total_ars,
                total_usdt: purchase.total_usdt,
            }),
        }
    }
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

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum RandomSelectionDto {
    Choices { values: Vec<String> },
    Range { start: String, end: String },
    Invalid,
}

impl From<RandomSelection> for RandomSelectionDto {
    fn from(value: RandomSelection) -> Self {
        match value {
            RandomSelection::Choices { values } => Self::Choices { values },
            RandomSelection::InclusiveRange { start, end } => Self::Range {
                start: start.to_string(),
                end: end.to_string(),
            },
            RandomSelection::Invalid => Self::Invalid,
        }
    }
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

/// Normalize adapter-preprocessed text into a Telegram slash command.
#[pyfunction]
fn normalize_command_text(message_text: &str) -> Option<String> {
    normalize_command_text_core(message_text)
}

/// Convert an arbitrary-precision number between bases from 2 through 36.
#[pyfunction]
fn convert_base(message_text: &str) -> PyResult<String> {
    let result = convert_base_core(message_text)
        .map_err(|_| PyValueError::new_err("Unicode input requires the legacy converter"))?;
    serde_json::to_string(&BaseConversionDto::from(result))
        .map_err(|error| PyValueError::new_err(format!("cannot encode base conversion: {error}")))
}

/// Parse a random choice or inclusive integer range without consuming randomness.
#[pyfunction]
fn parse_random_selection(message_text: &str) -> PyResult<String> {
    let result = parse_random_selection_core(message_text)
        .map_err(|_| PyValueError::new_err("Unicode range requires the legacy parser"))?;
    serde_json::to_string(&RandomSelectionDto::from(result))
        .map_err(|error| PyValueError::new_err(format!("cannot encode random selection: {error}")))
}

/// Map adapter-owned random samples to localization keys for a spontaneous reply.
#[pyfunction]
fn evaluate_random_reply(response_sample: i64, suffix_sample: i64) -> (&'static str, &'static str) {
    let result = evaluate_random_reply_core(response_sample, suffix_sample);
    let answer = match result.answer {
        RandomAnswer::Yes => "yes",
        RandomAnswer::No => "no",
    };
    let suffix = match result.suffix {
        RandomSuffix::None => "none",
        RandomSuffix::Address => "address",
        RandomSuffix::Name => "name",
    };
    (answer, suffix)
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

/// Evaluate and format one deterministic Bitcoin reference-price model.
#[pyfunction]
fn evaluate_market_model(model: &str, elapsed_days: i64, market_price: f64) -> PyResult<String> {
    let model = match model {
        "power_law" => MarketModel::PowerLaw,
        "rainbow" => MarketModel::Rainbow,
        _ => return Err(PyValueError::new_err("unknown market model")),
    };
    let result = evaluate_market_model_core(model, elapsed_days, market_price)
        .map_err(|_| PyValueError::new_err("elapsed days must be positive"))?;
    let value = MarketModelDto {
        value: format!("{:.2}", result.model_value),
        percentage: format!("{:.2}", result.percentage),
        valuation: match result.valuation {
            Valuation::Expensive => "expensive",
            Valuation::Cheap => "cheap",
        },
    };
    serde_json::to_string(&value)
        .map_err(|error| PyValueError::new_err(format!("cannot encode market model: {error}")))
}

/// Format two adapter-provided Bitcoin prices as the legacy satoshi command.
#[pyfunction]
fn format_satoshi_quote(price_usd: f64, price_ars: f64) -> PyResult<String> {
    format_satoshi_quote_core(price_usd, price_ars)
        .map_err(|_| PyValueError::new_err("Bitcoin prices must be finite and nonzero"))
}

/// Parse `/devo` input while preserving Python float compatibility.
#[pyfunction]
fn parse_devo_input(message_text: &str) -> PyResult<(String, f64, f64)> {
    let result = parse_devo_input_core(message_text)
        .map_err(|_| PyValueError::new_err("Unicode input requires the legacy parser"))?;
    Ok(match result {
        DevoInput::Valid { fee, purchase } => ("valid".to_owned(), fee, purchase),
        DevoInput::Usage => ("usage".to_owned(), 0.0, 0.0),
        DevoInput::InputError => ("input_error".to_owned(), 0.0, 0.0),
    })
}

/// Calculate `/devo` output values from normalized provider quotes.
#[pyfunction]
fn calculate_devo(
    fee: f64,
    purchase: f64,
    official: f64,
    card: f64,
    usdt_ask: f64,
    usdt_bid: f64,
) -> PyResult<String> {
    let result = calculate_devo_core(
        fee,
        purchase,
        DevoQuotes {
            official,
            card,
            usdt_ask,
            usdt_bid,
        },
    )
    .map_err(|_| PyValueError::new_err("Devo quotes would divide by zero"))?;
    serde_json::to_string(&DevoResultDto::from(result))
        .map_err(|error| PyValueError::new_err(format!("cannot encode devo result: {error}")))
}

/// Select and calculate all viable `/rulo` routes from normalized quotes.
#[pyfunction]
fn evaluate_rulo(input_json: &str) -> PyResult<String> {
    let input: RuloInputDto = serde_json::from_str(input_json)
        .map_err(|error| PyValueError::new_err(format!("invalid rulo input: {error}")))?;
    serde_json::to_string(&RuloEvaluationDto::from(evaluate_rulo_core(&input.into())))
        .map_err(|error| PyValueError::new_err(format!("cannot encode rulo result: {error}")))
}

/// Reconcile and rank adapter-provided Polymarket outcomes.
#[pyfunction]
fn rank_polymarket_outcomes(input_json: &str, limit: usize) -> PyResult<String> {
    let inputs: Vec<MarketOutcomeInputDto> = serde_json::from_str(input_json)
        .map_err(|error| PyValueError::new_err(format!("invalid Polymarket outcomes: {error}")))?;
    let outcomes = inputs
        .into_iter()
        .map(|input| MarketOutcome {
            title: input.title,
            cached_probability: input.cached_probability,
            live_probability: input.live_probability,
        })
        .collect::<Vec<_>>();
    let ranked = rank_outcomes_core(&outcomes, limit)
        .into_iter()
        .map(|outcome| RankedOutcomeDto {
            title: outcome.title,
            percentage: outcome.percentage,
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&ranked)
        .map_err(|error| PyValueError::new_err(format!("cannot encode ranked outcomes: {error}")))
}

/// Normalize one XML-adapter Hacker News item.
#[pyfunction]
fn normalize_hacker_news_item(title: &str, url: &str, description: &str) -> PyResult<String> {
    let item = normalize_hacker_news_item_core(title, url, description)
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .map(|item| HackerNewsItemDto {
            title: item.title,
            url: item.url,
            points: item.points,
            comments: item.comments,
            comments_url: item.comments_url,
        });
    serde_json::to_string(&item)
        .map_err(|error| PyValueError::new_err(format!("cannot encode Hacker News item: {error}")))
}

/// Format typed Hacker News items after Python localization.
#[pyfunction]
fn format_hacker_news_items(
    input_json: &str,
    include_discussion: bool,
    no_data: &str,
    comments_label: &str,
) -> PyResult<String> {
    let inputs: Vec<HackerNewsRenderItemDto> = serde_json::from_str(input_json)
        .map_err(|error| PyValueError::new_err(format!("invalid Hacker News items: {error}")))?;
    let items = inputs
        .into_iter()
        .map(|item| HackerNewsRenderItem {
            title: item.title,
            url: item.url,
            points: item.points,
            comments: item.comments,
            comments_url: item.comments_url,
        })
        .collect::<Vec<_>>();
    Ok(format_hacker_news_core(
        &items,
        include_discussion,
        no_data,
        comments_label,
    ))
}

/// Evaluate one typed chat-configuration callback transition.
#[pyfunction]
fn evaluate_config_callback(input_json: &str) -> PyResult<String> {
    let input: ConfigCallbackInputDto = serde_json::from_str(input_json)
        .map_err(|error| PyValueError::new_err(format!("invalid config callback: {error}")))?;
    let evaluation = evaluate_config_callback_core(&ConfigCallbackInput {
        action: input.action,
        value: input.value,
        current_toggle: input.current_toggle,
        current_creditless_limit: input.current_creditless_limit,
        numeric_value: input.numeric_value,
        timezone_min: input.timezone_min,
        timezone_max: input.timezone_max,
    });
    serde_json::to_string(&ConfigCallbackEvaluationDto::from(evaluation))
        .map_err(|error| PyValueError::new_err(format!("cannot encode config callback: {error}")))
}

/// Slice message text using Telegram UTF-16 entity offsets.
#[pyfunction]
fn slice_telegram_utf16(text: &str, offset: i64, length: i64) -> String {
    utf16_slice_core(text, offset, length)
}

/// Remove punctuation captured after a detected URL.
#[pyfunction]
fn trim_detected_url(raw_url: &str) -> String {
    trim_detected_url_core(raw_url)
}

/// Deduplicate detected URLs and apply the configured per-message limit.
#[pyfunction]
fn select_unique_urls(candidates: Vec<String>, max_links: usize) -> Vec<String> {
    select_unique_urls_core(&candidates, max_links)
}

/// Parse and clamp the optional `/creditlog` row limit.
#[pyfunction]
fn parse_creditlog_limit(message_text: &str) -> PyResult<Option<usize>> {
    match parse_creditlog_limit_core(message_text) {
        CreditLogLimit::Valid(limit) => Ok(Some(limit)),
        CreditLogLimit::Invalid => Ok(None),
        CreditLogLimit::NeedsLegacyParser => Err(PyValueError::new_err(
            "Unicode or underscored input requires the legacy parser",
        )),
    }
}

/// Truncate an admin report with a localized marker.
#[pyfunction]
fn truncate_admin_report(text: &str, max_length: usize, truncated_label: &str) -> String {
    truncate_admin_report_core(text, max_length, truncated_label)
}

/// Evaluate one stale-while-refresh cache entry.
#[pyfunction]
fn evaluate_cache_policy(
    cached_timestamp: Option<i64>,
    now: i64,
    ttl: i64,
    stale_grace: i64,
) -> &'static str {
    match evaluate_cache_core(cached_timestamp, now, ttl, stale_grace) {
        CacheDecision::Fresh => "fresh",
        CacheDecision::ServeStale => "stale",
        CacheDecision::RefreshInline => "refresh_inline",
    }
}

#[pyfunction]
fn request_cache_key(request_hash: &str) -> String {
    cache_key_core(request_hash)
}

#[pyfunction]
fn request_cache_history_key(hour_key: &str, request_hash: &str) -> String {
    cache_history_key_core(hour_key, request_hash)
}

#[pyfunction]
fn request_cache_ttl(expiration_time: i64) -> i64 {
    cache_ttl_core(expiration_time)
}

#[pyfunction]
fn last_success_ttl(ttl: i64, stale_grace: i64) -> i64 {
    last_success_ttl_core(ttl, stale_grace)
}

/// Prepare a versioned message entry and all keys for the existing atomic Redis write.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn prepare_message_write(
    chat_id: &str,
    message_id: &str,
    text: &str,
    timestamp: i64,
    role: Option<&str>,
    user_id: Option<&str>,
    username: Option<&str>,
    reply_to_message_id: Option<&str>,
    mentions_bot: bool,
) -> PyResult<String> {
    let plan = prepare_message_write_core(
        chat_id,
        message_id,
        text,
        timestamp,
        role,
        user_id,
        username,
        reply_to_message_id,
        mentions_bot,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    serde_json::to_string(&plan).map_err(|error| PyValueError::new_err(error.to_string()))
}

#[derive(Deserialize)]
struct SearchCandidateDto {
    message_id: String,
    text: String,
    reply_to_message_id: String,
    timestamp: i64,
}

#[pyfunction]
fn escape_message_search_text(query_text: &str) -> String {
    escape_search_text_core(query_text)
}

#[pyfunction]
fn escape_message_search_tag(value: &str) -> String {
    escape_search_tag_core(value)
}

#[pyfunction]
fn rank_message_search_results(
    candidates_json: &str,
    search_text: &str,
    reply_to_message_id: Option<&str>,
    excluded_message_ids: Vec<String>,
    limit: usize,
) -> PyResult<String> {
    let candidates = serde_json::from_str::<Vec<SearchCandidateDto>>(candidates_json)
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| SearchCandidate {
            index,
            message_id: candidate.message_id,
            text: candidate.text,
            reply_to_message_id: candidate.reply_to_message_id,
            timestamp: candidate.timestamp,
        })
        .collect::<Vec<_>>();
    let excluded = excluded_message_ids.into_iter().collect();
    serde_json::to_string(&rank_search_candidates_core(
        &candidates,
        search_text,
        reply_to_message_id,
        &excluded,
        limit,
    ))
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn message_state_key(kind: &str, chat_id: &str, message_id: Option<&str>) -> PyResult<String> {
    match kind {
        "summary" => Ok(summary_key_core(chat_id)),
        "user_summary" => Ok(user_summary_key_core(chat_id)),
        "compacted_until" => Ok(compacted_key_core(chat_id)),
        "user_compacted_until" => Ok(user_compacted_key_core(chat_id)),
        "bot_metadata" => message_id.map_or_else(
            || Err(PyValueError::new_err("bot metadata requires message_id")),
            |value| Ok(bot_metadata_key_core(chat_id, value)),
        ),
        "members" => Ok(members_key_core(chat_id)),
        _ => Err(PyValueError::new_err("unsupported message-state key kind")),
    }
}

#[pyfunction]
fn prepare_chat_member(first_name: &str, username: &str, last_seen: i64) -> PyResult<String> {
    prepare_chat_member_core(first_name, username, last_seen)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn evaluate_compaction_policy(
    current_summary: Option<&str>,
    current_marker: Option<&str>,
    prior_summary: Option<&str>,
    expected_marker: Option<&str>,
    result_summary: Option<&str>,
    target_marker: &str,
) -> &'static str {
    match evaluate_compaction_core(
        current_summary,
        current_marker,
        prior_summary,
        expected_marker,
        result_summary,
        target_marker,
    ) {
        CompactionDisposition::SettleRecoveredSuccess => "settle_recovered_success",
        CompactionDisposition::SettleObsolete => "settle_obsolete",
        CompactionDisposition::GenerateSummary => "generate_summary",
        CompactionDisposition::SaveAndSettle => "save_and_settle",
    }
}

#[pyfunction]
fn compaction_job_is_due(next_attempt_at: f64, now: f64) -> bool {
    compaction_is_due_core(next_attempt_at, now)
}

#[pyfunction]
fn compaction_retry_transition(
    attempts: u32,
    now: f64,
    has_billing_segment: bool,
) -> PyResult<String> {
    serde_json::to_string(&retry_compaction_core(attempts, now, has_billing_segment))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn normalize_compaction_job(payload: &str) -> PyResult<String> {
    normalize_compaction_job_adapter(payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn redis_media_cache_get(
    host: &str,
    port: u16,
    password: Option<&str>,
    prefix: &str,
    file_id: &str,
) -> PyResult<Option<String>> {
    get_cached_media_adapter(
        &RedisEndpoint {
            host: host.to_owned(),
            port,
            password: password.map(ToOwned::to_owned),
        },
        prefix,
        file_id,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn redis_media_cache_key(prefix: &str, file_id: &str) -> String {
    media_cache_key_adapter(prefix, file_id)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn redis_media_cache_set(
    host: &str,
    port: u16,
    password: Option<&str>,
    prefix: &str,
    file_id: &str,
    text: &str,
    ttl_seconds: i64,
) -> PyResult<()> {
    cache_media_adapter(
        &RedisEndpoint {
            host: host.to_owned(),
            port,
            password: password.map(ToOwned::to_owned),
        },
        prefix,
        file_id,
        text,
        ttl_seconds,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn redis_chat_admin_get(
    host: &str,
    port: u16,
    password: Option<&str>,
    chat_id: &str,
    user_id: &str,
) -> PyResult<Option<bool>> {
    get_cached_chat_admin_adapter(
        &RedisEndpoint {
            host: host.to_owned(),
            port,
            password: password.map(ToOwned::to_owned),
        },
        chat_id,
        user_id,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn redis_chat_admin_key(chat_id: &str, user_id: &str) -> String {
    chat_admin_cache_key_adapter(chat_id, user_id)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn redis_chat_admin_set(
    host: &str,
    port: u16,
    password: Option<&str>,
    chat_id: &str,
    user_id: &str,
    is_admin: bool,
    ttl_seconds: i64,
) -> PyResult<()> {
    cache_chat_admin_adapter(
        &RedisEndpoint {
            host: host.to_owned(),
            port,
            password: password.map(ToOwned::to_owned),
        },
        chat_id,
        user_id,
        is_admin,
        ttl_seconds,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Select one geocoding result from adapter-normalized qualifier keys.
#[pyfunction]
fn select_weather_location(
    qualifier_keys: Vec<String>,
    candidate_keys: Vec<String>,
) -> Option<usize> {
    select_location_candidate_core(&qualifier_keys, &candidate_keys)
}

/// Select one normalized hourly forecast row.
#[pyfunction]
fn select_weather_hour(
    forecast_hours: Vec<String>,
    provider_hour: Option<&str>,
    local_hour: &str,
) -> Option<usize> {
    select_forecast_hour_core(&forecast_hours, provider_hour, local_hour)
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
    module.add_function(wrap_pyfunction!(normalize_command_text, module)?)?;
    module.add_function(wrap_pyfunction!(convert_base, module)?)?;
    module.add_function(wrap_pyfunction!(parse_random_selection, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_random_reply, module)?)?;
    module.add_function(wrap_pyfunction!(parse_task_trigger, module)?)?;
    module.add_function(wrap_pyfunction!(parse_price_query, module)?)?;
    module.add_function(wrap_pyfunction!(format_market_info, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_market_model, module)?)?;
    module.add_function(wrap_pyfunction!(format_satoshi_quote, module)?)?;
    module.add_function(wrap_pyfunction!(parse_devo_input, module)?)?;
    module.add_function(wrap_pyfunction!(calculate_devo, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_rulo, module)?)?;
    module.add_function(wrap_pyfunction!(rank_polymarket_outcomes, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_hacker_news_item, module)?)?;
    module.add_function(wrap_pyfunction!(format_hacker_news_items, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_config_callback, module)?)?;
    module.add_function(wrap_pyfunction!(slice_telegram_utf16, module)?)?;
    module.add_function(wrap_pyfunction!(trim_detected_url, module)?)?;
    module.add_function(wrap_pyfunction!(select_unique_urls, module)?)?;
    module.add_function(wrap_pyfunction!(parse_creditlog_limit, module)?)?;
    module.add_function(wrap_pyfunction!(truncate_admin_report, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_cache_policy, module)?)?;
    module.add_function(wrap_pyfunction!(request_cache_key, module)?)?;
    module.add_function(wrap_pyfunction!(request_cache_history_key, module)?)?;
    module.add_function(wrap_pyfunction!(request_cache_ttl, module)?)?;
    module.add_function(wrap_pyfunction!(last_success_ttl, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_message_write, module)?)?;
    module.add_function(wrap_pyfunction!(escape_message_search_text, module)?)?;
    module.add_function(wrap_pyfunction!(escape_message_search_tag, module)?)?;
    module.add_function(wrap_pyfunction!(rank_message_search_results, module)?)?;
    module.add_function(wrap_pyfunction!(message_state_key, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_chat_member, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_compaction_policy, module)?)?;
    module.add_function(wrap_pyfunction!(compaction_job_is_due, module)?)?;
    module.add_function(wrap_pyfunction!(compaction_retry_transition, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_compaction_job, module)?)?;
    module.add_function(wrap_pyfunction!(redis_media_cache_get, module)?)?;
    module.add_function(wrap_pyfunction!(redis_media_cache_key, module)?)?;
    module.add_function(wrap_pyfunction!(redis_media_cache_set, module)?)?;
    module.add_function(wrap_pyfunction!(redis_chat_admin_get, module)?)?;
    module.add_function(wrap_pyfunction!(redis_chat_admin_key, module)?)?;
    module.add_function(wrap_pyfunction!(redis_chat_admin_set, module)?)?;
    module.add_function(wrap_pyfunction!(select_weather_location, module)?)?;
    module.add_function(wrap_pyfunction!(select_weather_hour, module)?)?;
    module.add_function(wrap_pyfunction!(should_auto_process_media, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_response_routing, module)?)?;
    Ok(())
}
