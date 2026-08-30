//! Validation and normalization for scheduled-task triggers.

/// A dynamically supplied integer after boundary type classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegerInput {
    /// The value was absent or null.
    Missing,
    /// The value was present but was not an integer.
    Invalid,
    /// The integer was below the bridge's representable range.
    BelowRange,
    /// The integer was above the bridge's representable range.
    AboveRange,
    /// A representable integer value.
    Value(i64),
}

/// Typed trigger configuration supplied by an external adapter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TriggerConfigInput {
    /// No mapping was supplied.
    Missing,
    /// A mapping with an unsupported trigger type was supplied.
    Unsupported,
    /// A cron trigger configuration.
    Cron {
        /// Hour input.
        hour: IntegerInput,
        /// Minute input.
        minute: IntegerInput,
        /// Optional comma-separated weekday input.
        weekdays: Option<String>,
        /// Optional day-of-month input.
        day: IntegerInput,
    },
    /// A day-based interval configuration.
    IntervalDays {
        /// Day count input.
        days: IntegerInput,
    },
}

/// All inputs accepted by task-trigger parsing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaskTriggerInput {
    /// One-shot delay. A present value takes priority over every other input.
    pub delay_seconds: IntegerInput,
    /// Repeating second interval. A present value takes priority over config.
    pub interval_seconds: IntegerInput,
    /// Cron or day-interval configuration.
    pub config: TriggerConfigInput,
}

/// A validated scheduled-task trigger.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TaskTrigger {
    /// Run once after a delay.
    Delay { seconds: i64 },
    /// Repeat after a number of seconds.
    IntervalSeconds { seconds: i64 },
    /// Repeat after a number of days.
    IntervalDays { days: i64 },
    /// Run on a cron-like calendar schedule.
    Cron {
        /// Hour from 0 through 23.
        hour: i64,
        /// Minute from 0 through 59.
        minute: i64,
        /// Normalized English weekday abbreviations.
        weekdays: Vec<String>,
        /// Optional day of the month.
        day: Option<i64>,
    },
}

/// Stable validation errors localized by the caller.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TriggerError {
    Required,
    UnsupportedType,
    DelayPositive,
    DelayMaximum,
    IntervalMinimum,
    IntervalMaximum,
    DaysRequired,
    DaysPositive,
    DaysMaximum,
    HourRequired,
    HourRange,
    MinuteRequired,
    MinuteRange,
    Weekday { value: String },
    WeekdayEmpty,
    DayRange,
}

const MAX_DELAY_SECONDS: i64 = 86_400 * 3_650;
const MIN_INTERVAL_SECONDS: i64 = 300;
const MAX_INTERVAL_SECONDS: i64 = 86_400 * 7;
const MAX_INTERVAL_DAYS: i64 = 90;

/// Validate and normalize one task trigger with the legacy input precedence.
pub fn parse_task_trigger(input: TaskTriggerInput) -> Result<TaskTrigger, TriggerError> {
    if input.delay_seconds != IntegerInput::Missing {
        return parse_delay(input.delay_seconds);
    }
    if input.interval_seconds != IntegerInput::Missing {
        return parse_interval_seconds(input.interval_seconds);
    }

    match input.config {
        TriggerConfigInput::Missing => Err(TriggerError::Required),
        TriggerConfigInput::Unsupported => Err(TriggerError::UnsupportedType),
        TriggerConfigInput::IntervalDays { days } => parse_interval_days(days),
        TriggerConfigInput::Cron {
            hour,
            minute,
            weekdays,
            day,
        } => parse_cron(hour, minute, weekdays.as_deref(), day),
    }
}

fn parse_delay(input: IntegerInput) -> Result<TaskTrigger, TriggerError> {
    match input {
        IntegerInput::Value(seconds) if seconds > MAX_DELAY_SECONDS => {
            Err(TriggerError::DelayMaximum)
        }
        IntegerInput::AboveRange => Err(TriggerError::DelayMaximum),
        IntegerInput::Value(seconds) if seconds >= 1 => Ok(TaskTrigger::Delay { seconds }),
        IntegerInput::Missing
        | IntegerInput::Invalid
        | IntegerInput::BelowRange
        | IntegerInput::Value(_) => Err(TriggerError::DelayPositive),
    }
}

fn parse_interval_seconds(input: IntegerInput) -> Result<TaskTrigger, TriggerError> {
    match input {
        IntegerInput::Value(seconds) if seconds > MAX_INTERVAL_SECONDS => {
            Err(TriggerError::IntervalMaximum)
        }
        IntegerInput::AboveRange => Err(TriggerError::IntervalMaximum),
        IntegerInput::Value(seconds) if seconds >= MIN_INTERVAL_SECONDS => {
            Ok(TaskTrigger::IntervalSeconds { seconds })
        }
        IntegerInput::Missing
        | IntegerInput::Invalid
        | IntegerInput::BelowRange
        | IntegerInput::Value(_) => Err(TriggerError::IntervalMinimum),
    }
}

fn parse_interval_days(input: IntegerInput) -> Result<TaskTrigger, TriggerError> {
    match input {
        IntegerInput::Missing => Err(TriggerError::DaysRequired),
        IntegerInput::Value(days) if days > MAX_INTERVAL_DAYS => Err(TriggerError::DaysMaximum),
        IntegerInput::AboveRange => Err(TriggerError::DaysMaximum),
        IntegerInput::Value(days) if days >= 1 => Ok(TaskTrigger::IntervalDays { days }),
        IntegerInput::Invalid | IntegerInput::BelowRange | IntegerInput::Value(_) => {
            Err(TriggerError::DaysPositive)
        }
    }
}

fn required_bounded_integer(
    input: IntegerInput,
    minimum: i64,
    maximum: i64,
    missing_error: TriggerError,
    range_error: TriggerError,
) -> Result<i64, TriggerError> {
    match input {
        IntegerInput::Missing => Err(missing_error),
        IntegerInput::Value(value) if (minimum..=maximum).contains(&value) => Ok(value),
        IntegerInput::Invalid
        | IntegerInput::BelowRange
        | IntegerInput::AboveRange
        | IntegerInput::Value(_) => Err(range_error),
    }
}

fn parse_cron(
    hour_input: IntegerInput,
    minute_input: IntegerInput,
    weekday_input: Option<&str>,
    day_input: IntegerInput,
) -> Result<TaskTrigger, TriggerError> {
    let hour = required_bounded_integer(
        hour_input,
        0,
        23,
        TriggerError::HourRequired,
        TriggerError::HourRange,
    )?;
    let minute = required_bounded_integer(
        minute_input,
        0,
        59,
        TriggerError::MinuteRequired,
        TriggerError::MinuteRange,
    )?;
    let weekdays = parse_weekdays(weekday_input)?;
    let day = match day_input {
        IntegerInput::Missing => None,
        IntegerInput::Value(value) if (1..=31).contains(&value) => Some(value),
        IntegerInput::Invalid
        | IntegerInput::BelowRange
        | IntegerInput::AboveRange
        | IntegerInput::Value(_) => return Err(TriggerError::DayRange),
    };

    Ok(TaskTrigger::Cron {
        hour,
        minute,
        weekdays,
        day,
    })
}

fn parse_weekdays(input: Option<&str>) -> Result<Vec<String>, TriggerError> {
    let Some(input) = input else {
        return Ok(Vec::new());
    };
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut weekdays = Vec::new();
    for part in input.split(',') {
        let token = part.trim().to_lowercase();
        if token.is_empty() {
            continue;
        }
        let normalized = match token.as_str() {
            "lun" => "mon",
            "mar" => "tue",
            "mie" => "wed",
            "jue" => "thu",
            "vie" => "fri",
            "sab" => "sat",
            "dom" => "sun",
            "mon" | "tue" | "wed" | "thu" | "fri" | "sat" | "sun" => token.as_str(),
            _ => return Err(TriggerError::Weekday { value: token }),
        };
        weekdays.push(normalized.to_owned());
    }

    if weekdays.is_empty() {
        Err(TriggerError::WeekdayEmpty)
    } else {
        Ok(weekdays)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        IntegerInput, TaskTrigger, TaskTriggerInput, TriggerConfigInput, TriggerError,
        parse_task_trigger,
    };

    fn input(config: TriggerConfigInput) -> TaskTriggerInput {
        TaskTriggerInput {
            delay_seconds: IntegerInput::Missing,
            interval_seconds: IntegerInput::Missing,
            config,
        }
    }

    #[test]
    fn delay_has_priority_and_enforces_bounds() {
        let mut value = input(TriggerConfigInput::Unsupported);
        value.delay_seconds = IntegerInput::Value(30);
        value.interval_seconds = IntegerInput::Value(600);
        assert_eq!(
            parse_task_trigger(value),
            Ok(TaskTrigger::Delay { seconds: 30 })
        );

        let mut too_large = input(TriggerConfigInput::Missing);
        too_large.delay_seconds = IntegerInput::AboveRange;
        assert_eq!(
            parse_task_trigger(too_large),
            Err(TriggerError::DelayMaximum)
        );
    }

    #[test]
    fn second_intervals_enforce_both_bounds() {
        let mut too_short = input(TriggerConfigInput::Missing);
        too_short.interval_seconds = IntegerInput::Value(299);
        assert_eq!(
            parse_task_trigger(too_short),
            Err(TriggerError::IntervalMinimum)
        );

        let mut valid = input(TriggerConfigInput::Missing);
        valid.interval_seconds = IntegerInput::Value(300);
        assert_eq!(
            parse_task_trigger(valid),
            Ok(TaskTrigger::IntervalSeconds { seconds: 300 })
        );
    }

    #[test]
    fn day_intervals_distinguish_missing_invalid_and_excessive_values() {
        for (days, expected) in [
            (IntegerInput::Missing, TriggerError::DaysRequired),
            (IntegerInput::Invalid, TriggerError::DaysPositive),
            (IntegerInput::Value(0), TriggerError::DaysPositive),
            (IntegerInput::Value(91), TriggerError::DaysMaximum),
        ] {
            assert_eq!(
                parse_task_trigger(input(TriggerConfigInput::IntervalDays { days })),
                Err(expected)
            );
        }
    }

    #[test]
    fn cron_normalizes_weekdays_and_preserves_day() {
        assert_eq!(
            parse_task_trigger(input(TriggerConfigInput::Cron {
                hour: IntegerInput::Value(9),
                minute: IntegerInput::Value(5),
                weekdays: Some(" lun, WED ".to_owned()),
                day: IntegerInput::Value(12),
            })),
            Ok(TaskTrigger::Cron {
                hour: 9,
                minute: 5,
                weekdays: vec!["mon".to_owned(), "wed".to_owned()],
                day: Some(12),
            })
        );
    }

    #[test]
    fn cron_reports_fields_in_legacy_validation_order() {
        let config = TriggerConfigInput::Cron {
            hour: IntegerInput::Missing,
            minute: IntegerInput::Missing,
            weekdays: Some("bad".to_owned()),
            day: IntegerInput::Value(99),
        };
        assert_eq!(
            parse_task_trigger(input(config)),
            Err(TriggerError::HourRequired)
        );
    }

    #[test]
    fn cron_rejects_invalid_and_empty_weekday_lists() {
        let make_config = |weekdays| TriggerConfigInput::Cron {
            hour: IntegerInput::Value(0),
            minute: IntegerInput::Value(0),
            weekdays: Some(weekdays),
            day: IntegerInput::Missing,
        };
        assert_eq!(
            parse_task_trigger(input(make_config("foo".to_owned()))),
            Err(TriggerError::Weekday {
                value: "foo".to_owned()
            })
        );
        assert_eq!(
            parse_task_trigger(input(make_config(" , ".to_owned()))),
            Err(TriggerError::WeekdayEmpty)
        );
    }

    #[test]
    fn missing_and_unsupported_configs_have_distinct_errors() {
        assert_eq!(
            parse_task_trigger(input(TriggerConfigInput::Missing)),
            Err(TriggerError::Required)
        );
        assert_eq!(
            parse_task_trigger(input(TriggerConfigInput::Unsupported)),
            Err(TriggerError::UnsupportedType)
        );
    }
}
