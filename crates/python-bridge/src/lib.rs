//! Temporary Python bridge for incrementally adopting `bot-core`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use bot_core::command_parsing::parse_command as parse_command_core;
use bot_core::credit_units::{
    CreditUnits, format_credit_units as format_credit_units_core,
    parse_credit_units as parse_credit_units_core,
    rescale_credit_units as rescale_credit_units_core,
    whole_credits_to_units as whole_credits_to_units_core,
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

/// Register the temporary `respondedorbot_rs` Python module.
#[pymodule]
fn respondedorbot_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(migration_protocol_version, module)?)?;
    module.add_function(wrap_pyfunction!(whole_credits_to_units, module)?)?;
    module.add_function(wrap_pyfunction!(rescale_credit_units, module)?)?;
    module.add_function(wrap_pyfunction!(parse_credit_units, module)?)?;
    module.add_function(wrap_pyfunction!(format_credit_units, module)?)?;
    module.add_function(wrap_pyfunction!(parse_command, module)?)?;
    module.add_function(wrap_pyfunction!(parse_task_trigger, module)?)?;
    Ok(())
}
