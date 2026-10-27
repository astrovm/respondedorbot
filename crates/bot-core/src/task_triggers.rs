//! Validation and normalization for scheduled-task triggers.

/// A dynamically supplied integer after boundary type classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegerInput {
    /// The value was absent or null.
    Missing,
    /// The value was present but was not an integer.
    Invalid,
    /// The integer was below the supported range.
    BelowRange,
    /// The integer was above the supported range.
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
    /// A mapping with a field its trigger type does not use.
    UnknownField {
        /// The unused field name.
        name: String,
    },
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
    /// A repeating interval split into units that are added together.
    Interval {
        /// Day count input.
        days: IntegerInput,
        /// Hour count input.
        hours: IntegerInput,
        /// Minute count input.
        minutes: IntegerInput,
        /// Second count input.
        seconds: IntegerInput,
    },
}

/// All inputs accepted by task-trigger parsing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaskTriggerInput {
    /// One-shot delay. A present value takes priority over every other input.
    pub delay_seconds: IntegerInput,
    /// Repeating second interval. A present value takes priority over config.
    pub interval_seconds: IntegerInput,
    /// Cron or unit-interval configuration.
    pub config: TriggerConfigInput,
}

/// A validated scheduled-task trigger.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TaskTrigger {
    /// Run once after a delay.
    Delay { seconds: i64 },
    /// Repeat after a number of seconds.
    IntervalSeconds { seconds: i64 },
    /// Repeat after a whole number of days longer than a week.
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
    UnknownField { name: String },
    DelayPositive,
    DelayMaximum,
    IntervalMinimum,
    IntervalMaximum,
    IntervalWholeDays,
    IntervalUnitsRequired,
    IntervalUnitInvalid,
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
/// Longer intervals are stored as whole days.
const MAX_SECOND_INTERVAL: i64 = 86_400 * 7;
const MAX_INTERVAL_SECONDS: i64 = 86_400 * 90;

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
        TriggerConfigInput::UnknownField { name } => Err(TriggerError::UnknownField { name }),
        TriggerConfigInput::Interval {
            days,
            hours,
            minutes,
            seconds,
        } => parse_interval_units([(days, 86_400), (hours, 3_600), (minutes, 60), (seconds, 1)]),
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
        IntegerInput::Value(seconds) if seconds > MAX_SECOND_INTERVAL => {
            if seconds % 86_400 == 0 {
                Ok(TaskTrigger::IntervalDays {
                    days: seconds / 86_400,
                })
            } else {
                Err(TriggerError::IntervalWholeDays)
            }
        }
        IntegerInput::Value(seconds) if seconds >= MIN_INTERVAL_SECONDS => {
            Ok(TaskTrigger::IntervalSeconds { seconds })
        }
        IntegerInput::Missing
        | IntegerInput::Invalid
        | IntegerInput::BelowRange
        | IntegerInput::Value(_) => Err(TriggerError::IntervalMinimum),
    }
}

/// Add up `(amount, seconds per unit)` pairs into one interval.
fn parse_interval_units(units: [(IntegerInput, i64); 4]) -> Result<TaskTrigger, TriggerError> {
    if units
        .iter()
        .all(|(amount, _)| *amount == IntegerInput::Missing)
    {
        return Err(TriggerError::IntervalUnitsRequired);
    }
    let mut total: i64 = 0;
    for (amount, unit_seconds) in units {
        let seconds = match amount {
            IntegerInput::Missing => 0,
            IntegerInput::AboveRange => return Err(TriggerError::IntervalMaximum),
            IntegerInput::Value(amount) if amount >= 0 => amount
                .checked_mul(unit_seconds)
                .ok_or(TriggerError::IntervalMaximum)?,
            IntegerInput::Invalid | IntegerInput::BelowRange | IntegerInput::Value(_) => {
                return Err(TriggerError::IntervalUnitInvalid);
            }
        };
        total = total
            .checked_add(seconds)
            .ok_or(TriggerError::IntervalMaximum)?;
    }
    parse_interval_seconds(IntegerInput::Value(total))
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

    fn units(days: i64, hours: i64, minutes: i64) -> TriggerConfigInput {
        let unit = |value| {
            if value == 0 {
                IntegerInput::Missing
            } else {
                IntegerInput::Value(value)
            }
        };
        TriggerConfigInput::Interval {
            days: unit(days),
            hours: unit(hours),
            minutes: unit(minutes),
            seconds: IntegerInput::Missing,
        }
    }

    #[test]
    fn unit_intervals_add_up_hours_minutes_and_days() {
        assert_eq!(
            parse_task_trigger(input(units(0, 3, 0))),
            Ok(TaskTrigger::IntervalSeconds { seconds: 10_800 })
        );
        assert_eq!(
            parse_task_trigger(input(units(1, 2, 30))),
            Ok(TaskTrigger::IntervalSeconds { seconds: 95_400 })
        );
        assert_eq!(
            parse_task_trigger(input(units(30, 0, 0))),
            Ok(TaskTrigger::IntervalDays { days: 30 })
        );
        assert_eq!(
            parse_task_trigger(input(TriggerConfigInput::Interval {
                days: IntegerInput::Value(0),
                hours: IntegerInput::Missing,
                minutes: IntegerInput::Missing,
                seconds: IntegerInput::Value(600),
            })),
            Ok(TaskTrigger::IntervalSeconds { seconds: 600 })
        );
    }

    #[test]
    fn unit_intervals_distinguish_missing_invalid_and_excessive_values() {
        let days = |days| TriggerConfigInput::Interval {
            days,
            hours: IntegerInput::Missing,
            minutes: IntegerInput::Missing,
            seconds: IntegerInput::Missing,
        };
        for (config, expected) in [
            (
                days(IntegerInput::Missing),
                TriggerError::IntervalUnitsRequired,
            ),
            (
                days(IntegerInput::Invalid),
                TriggerError::IntervalUnitInvalid,
            ),
            (
                days(IntegerInput::BelowRange),
                TriggerError::IntervalUnitInvalid,
            ),
            (
                days(IntegerInput::Value(-1)),
                TriggerError::IntervalUnitInvalid,
            ),
            (days(IntegerInput::Value(0)), TriggerError::IntervalMinimum),
            (days(IntegerInput::Value(91)), TriggerError::IntervalMaximum),
            (
                days(IntegerInput::AboveRange),
                TriggerError::IntervalMaximum,
            ),
            (
                days(IntegerInput::Value(i64::MAX)),
                TriggerError::IntervalMaximum,
            ),
            (units(0, 0, 4), TriggerError::IntervalMinimum),
            (units(8, 1, 0), TriggerError::IntervalWholeDays),
        ] {
            assert_eq!(parse_task_trigger(input(config)), Err(expected));
        }
        assert_eq!(
            parse_task_trigger(input(TriggerConfigInput::Interval {
                days: IntegerInput::Value(i64::MAX / 86_400),
                hours: IntegerInput::Missing,
                minutes: IntegerInput::Missing,
                seconds: IntegerInput::Value(i64::MAX),
            })),
            Err(TriggerError::IntervalMaximum)
        );
    }

    #[test]
    fn unknown_config_fields_are_reported_by_name() {
        assert_eq!(
            parse_task_trigger(input(TriggerConfigInput::UnknownField {
                name: "weekdays".to_owned()
            })),
            Err(TriggerError::UnknownField {
                name: "weekdays".to_owned()
            })
        );
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

    #[test]
    fn delays_and_intervals_reject_every_out_of_range_shape() {
        let delay = |delay_seconds| TaskTriggerInput {
            delay_seconds,
            ..input(TriggerConfigInput::Missing)
        };
        assert_eq!(
            parse_task_trigger(delay(IntegerInput::Value(86_400 * 3_650 + 1))),
            Err(TriggerError::DelayMaximum)
        );
        for below in [
            IntegerInput::Value(0),
            IntegerInput::BelowRange,
            IntegerInput::Invalid,
        ] {
            assert_eq!(
                parse_task_trigger(delay(below)),
                Err(TriggerError::DelayPositive)
            );
        }
        let interval = |interval_seconds| TaskTriggerInput {
            interval_seconds,
            ..input(TriggerConfigInput::Missing)
        };
        for above in [
            IntegerInput::Value(86_400 * 90 + 1),
            IntegerInput::Value(86_400 * 91),
            IntegerInput::AboveRange,
        ] {
            assert_eq!(
                parse_task_trigger(interval(above)),
                Err(TriggerError::IntervalMaximum)
            );
        }
        assert_eq!(
            parse_task_trigger(interval(IntegerInput::Value(86_400 * 7 + 1))),
            Err(TriggerError::IntervalWholeDays)
        );
        for invalid in [IntegerInput::Invalid, IntegerInput::BelowRange] {
            assert_eq!(
                parse_task_trigger(interval(invalid)),
                Err(TriggerError::IntervalMinimum)
            );
        }
    }

    #[test]
    fn cron_fields_report_their_own_errors_and_accept_every_weekday_alias() {
        let cron = |hour, minute, weekdays: Option<&str>, day| {
            parse_task_trigger(input(TriggerConfigInput::Cron {
                hour,
                minute,
                weekdays: weekdays.map(str::to_owned),
                day,
            }))
        };
        let ok = IntegerInput::Value(1);
        assert_eq!(
            cron(IntegerInput::Invalid, ok, None, IntegerInput::Missing),
            Err(TriggerError::HourRange)
        );
        assert_eq!(
            cron(ok, IntegerInput::Missing, None, IntegerInput::Missing),
            Err(TriggerError::MinuteRequired)
        );
        assert_eq!(
            cron(ok, IntegerInput::Value(60), None, IntegerInput::Missing),
            Err(TriggerError::MinuteRange)
        );
        assert_eq!(
            cron(ok, ok, Some("xyz"), IntegerInput::Missing),
            Err(TriggerError::Weekday {
                value: "xyz".to_owned()
            })
        );
        assert_eq!(
            cron(ok, ok, None, IntegerInput::Value(32)),
            Err(TriggerError::DayRange)
        );
        assert_eq!(
            cron(ok, ok, Some(""), IntegerInput::Missing),
            Ok(TaskTrigger::Cron {
                hour: 1,
                minute: 1,
                weekdays: Vec::new(),
                day: None,
            })
        );
        assert_eq!(
            cron(
                ok,
                ok,
                Some("lun,mar,mie,jue,vie,sab,dom,Sun"),
                IntegerInput::Missing
            ),
            Ok(TaskTrigger::Cron {
                hour: 1,
                minute: 1,
                weekdays: ["mon", "tue", "wed", "thu", "fri", "sat", "sun", "sun"]
                    .map(str::to_owned)
                    .to_vec(),
                day: None,
            })
        );
    }

    #[test]
    fn delays_and_intervals_accept_their_exact_maximum() {
        let delay = |seconds: i64| {
            parse_task_trigger(TaskTriggerInput {
                delay_seconds: IntegerInput::Value(seconds),
                ..input(TriggerConfigInput::Missing)
            })
        };
        assert_eq!(
            delay(315_360_000),
            Ok(TaskTrigger::Delay {
                seconds: 315_360_000
            })
        );
        assert_eq!(delay(315_360_001), Err(TriggerError::DelayMaximum));
        let interval = |seconds: i64| {
            parse_task_trigger(TaskTriggerInput {
                interval_seconds: IntegerInput::Value(seconds),
                ..input(TriggerConfigInput::Missing)
            })
        };
        assert_eq!(
            interval(604_800),
            Ok(TaskTrigger::IntervalSeconds { seconds: 604_800 })
        );
        assert_eq!(interval(691_200), Ok(TaskTrigger::IntervalDays { days: 8 }));
        assert_eq!(
            interval(7_776_000),
            Ok(TaskTrigger::IntervalDays { days: 90 })
        );
        assert_eq!(interval(7_776_001), Err(TriggerError::IntervalMaximum));
    }

    proptest::proptest! {
        #[test]
        fn delays_are_valid_from_one_second_to_ten_years(seconds in proptest::prelude::any::<i64>()) {
            let parsed = parse_task_trigger(TaskTriggerInput {
                delay_seconds: IntegerInput::Value(seconds),
                ..input(TriggerConfigInput::Missing)
            });
            let expected = match seconds {
                1..=315_360_000 => Ok(TaskTrigger::Delay { seconds }),
                ..=0 => Err(TriggerError::DelayPositive),
                _ => Err(TriggerError::DelayMaximum),
            };
            proptest::prop_assert_eq!(parsed, expected);
        }

        #[test]
        fn intervals_are_valid_from_five_minutes_to_ninety_days(seconds in proptest::prelude::any::<i64>()) {
            let parsed = parse_task_trigger(TaskTriggerInput {
                interval_seconds: IntegerInput::Value(seconds),
                ..input(TriggerConfigInput::Missing)
            });
            let expected = match seconds {
                ..=299 => Err(TriggerError::IntervalMinimum),
                300..=604_800 => Ok(TaskTrigger::IntervalSeconds { seconds }),
                604_801..=7_776_000 if seconds % 86_400 == 0 => {
                    Ok(TaskTrigger::IntervalDays { days: seconds / 86_400 })
                }
                604_801..=7_776_000 => Err(TriggerError::IntervalWholeDays),
                _ => Err(TriggerError::IntervalMaximum),
            };
            proptest::prop_assert_eq!(parsed, expected);
        }

        #[test]
        fn unit_intervals_match_the_same_total_in_seconds(
            days in 0_i64..200,
            hours in 0_i64..5_000,
            minutes in 0_i64..300_000,
        ) {
            let by_units = parse_task_trigger(input(TriggerConfigInput::Interval {
                days: IntegerInput::Value(days),
                hours: IntegerInput::Value(hours),
                minutes: IntegerInput::Value(minutes),
                seconds: IntegerInput::Missing,
            }));
            let by_seconds = parse_task_trigger(TaskTriggerInput {
                interval_seconds: IntegerInput::Value(days * 86_400 + hours * 3_600 + minutes * 60),
                ..input(TriggerConfigInput::Missing)
            });
            proptest::prop_assert_eq!(by_units, by_seconds);
        }
    }
}
