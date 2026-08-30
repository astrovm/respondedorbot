//! Canonical scheduled-task state and deterministic recurrence decisions.

use chrono::{DateTime, Datelike, FixedOffset, TimeZone, Utc, Weekday};
use thiserror::Error;

/// Canonical task payload version written during the compatibility migration.
pub const TASK_SCHEMA_VERSION: u8 = 1;
/// APScheduler's existing late-execution allowance.
pub const MISFIRE_GRACE_SECONDS: i64 = 300;

/// Stable task identifier used in Redis keys and callback data.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct TaskId(String);

impl TaskId {
    pub fn new(value: impl Into<String>) -> Result<Self, TaskStateError> {
        let value = value.into();
        if value.is_empty() || value.len() > 128 {
            return Err(TaskStateError::InvalidTaskId);
        }
        if !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(TaskStateError::InvalidTaskId);
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Language-neutral schedule reconstructed from the legacy trigger fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TaskSchedule {
    Once,
    IntervalSeconds {
        seconds: i64,
    },
    IntervalDays {
        days: i64,
    },
    Cron {
        hour: u32,
        minute: u32,
        weekdays: Vec<Weekday>,
        day: Option<u32>,
    },
}

impl TaskSchedule {
    #[must_use]
    pub const fn is_recurring(&self) -> bool {
        !matches!(self, Self::Once)
    }
}

/// Canonical task data after adapter validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledTask {
    pub id: TaskId,
    pub chat_id: String,
    pub text: String,
    pub user_name: String,
    pub user_id: Option<i64>,
    pub schedule: TaskSchedule,
    pub timezone_offset: i32,
    pub locale: String,
    pub schedule_anchor_at: Option<i64>,
    pub next_run_at: Option<i64>,
    pub last_execution_id: Option<String>,
}

/// Action selected for one scheduler observation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DueDecision {
    Wait,
    Skip {
        remove_task: bool,
        next_run_at: Option<i64>,
    },
    Execute {
        execution_id: String,
        scheduled_for: i64,
        next_run_at: Option<i64>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum TaskStateError {
    #[error("task id is invalid")]
    InvalidTaskId,
    #[error("task schedule has an invalid interval")]
    InvalidInterval,
    #[error("task schedule has an invalid timezone offset")]
    InvalidTimezone,
    #[error("task schedule has an invalid cron field")]
    InvalidCron,
    #[error("task timestamp is outside the supported range")]
    InvalidTimestamp,
    #[error("no matching cron occurrence exists in the supported range")]
    CronSearchExhausted,
}

fn utc_timestamp(value: i64) -> Result<DateTime<Utc>, TaskStateError> {
    Utc.timestamp_opt(value, 0)
        .single()
        .ok_or(TaskStateError::InvalidTimestamp)
}

fn fixed_offset(hours: i32) -> Result<FixedOffset, TaskStateError> {
    if !(-12..=14).contains(&hours) {
        return Err(TaskStateError::InvalidTimezone);
    }
    hours
        .checked_mul(3_600)
        .and_then(FixedOffset::east_opt)
        .ok_or(TaskStateError::InvalidTimezone)
}

fn interval_seconds(schedule: &TaskSchedule) -> Result<Option<i64>, TaskStateError> {
    match schedule {
        TaskSchedule::Once | TaskSchedule::Cron { .. } => Ok(None),
        TaskSchedule::IntervalSeconds { seconds } if *seconds > 0 => Ok(Some(*seconds)),
        TaskSchedule::IntervalDays { days } if *days > 0 => days
            .checked_mul(86_400)
            .map(Some)
            .ok_or(TaskStateError::InvalidInterval),
        TaskSchedule::IntervalSeconds { .. } | TaskSchedule::IntervalDays { .. } => {
            Err(TaskStateError::InvalidInterval)
        }
    }
}

fn cron_matches(
    schedule: &TaskSchedule,
    weekday: Weekday,
    day_of_month: u32,
) -> Result<bool, TaskStateError> {
    let TaskSchedule::Cron {
        hour,
        minute,
        weekdays,
        day,
    } = schedule
    else {
        return Err(TaskStateError::InvalidCron);
    };
    if *hour > 23 || *minute > 59 || day.is_some_and(|value| !(1..=31).contains(&value)) {
        return Err(TaskStateError::InvalidCron);
    }
    Ok((weekdays.is_empty() || weekdays.contains(&weekday))
        && day.is_none_or(|expected| expected == day_of_month))
}

fn next_cron_occurrence(
    schedule: &TaskSchedule,
    timezone_offset: i32,
    after: i64,
) -> Result<i64, TaskStateError> {
    let TaskSchedule::Cron { hour, minute, .. } = schedule else {
        return Err(TaskStateError::InvalidCron);
    };
    if *hour > 23 || *minute > 59 {
        return Err(TaskStateError::InvalidCron);
    }
    let offset = fixed_offset(timezone_offset)?;
    let local_after = utc_timestamp(after)?.with_timezone(&offset);
    let mut date = local_after.date_naive();

    // Ten years covers every supported monthly schedule while keeping malformed
    // calendar combinations bounded.
    for _ in 0..=3_660 {
        let naive = date
            .and_hms_opt(*hour, *minute, 0)
            .ok_or(TaskStateError::InvalidCron)?;
        let candidate = offset
            .from_local_datetime(&naive)
            .single()
            .ok_or(TaskStateError::InvalidTimestamp)?;
        if candidate > local_after && cron_matches(schedule, candidate.weekday(), candidate.day())?
        {
            return Ok(candidate.with_timezone(&Utc).timestamp());
        }
        date = date.succ_opt().ok_or(TaskStateError::CronSearchExhausted)?;
    }
    Err(TaskStateError::CronSearchExhausted)
}

/// Compute the first occurrence after task creation.
pub fn initial_next_run(
    schedule: &TaskSchedule,
    timezone_offset: i32,
    created_at: i64,
    delay_seconds: Option<i64>,
) -> Result<i64, TaskStateError> {
    if let Some(interval) = interval_seconds(schedule)? {
        return created_at
            .checked_add(interval)
            .ok_or(TaskStateError::InvalidTimestamp);
    }
    match schedule {
        TaskSchedule::Once => created_at
            .checked_add(
                delay_seconds
                    .filter(|value| *value > 0)
                    .ok_or(TaskStateError::InvalidInterval)?,
            )
            .ok_or(TaskStateError::InvalidTimestamp),
        TaskSchedule::Cron { .. } => next_cron_occurrence(schedule, timezone_offset, created_at),
        TaskSchedule::IntervalSeconds { .. } | TaskSchedule::IntervalDays { .. } => {
            Err(TaskStateError::InvalidInterval)
        }
    }
}

/// Find the first recurring occurrence strictly after `after` while preserving
/// the current occurrence as the interval anchor.
pub fn next_run_after(
    schedule: &TaskSchedule,
    timezone_offset: i32,
    current_run_at: i64,
    after: i64,
) -> Result<Option<i64>, TaskStateError> {
    if let Some(interval) = interval_seconds(schedule)? {
        let elapsed = after.saturating_sub(current_run_at);
        let steps = elapsed.div_euclid(interval).saturating_add(1);
        return current_run_at
            .checked_add(
                interval
                    .checked_mul(steps)
                    .ok_or(TaskStateError::InvalidTimestamp)?,
            )
            .map(Some)
            .ok_or(TaskStateError::InvalidTimestamp);
    }
    match schedule {
        TaskSchedule::Once => Ok(None),
        TaskSchedule::Cron { .. } => {
            next_cron_occurrence(schedule, timezone_offset, after).map(Some)
        }
        TaskSchedule::IntervalSeconds { .. } | TaskSchedule::IntervalDays { .. } => {
            Err(TaskStateError::InvalidInterval)
        }
    }
}

/// Apply APScheduler-compatible grace and coalescing rules to one task.
pub fn evaluate_due(task: &ScheduledTask, now: i64) -> Result<DueDecision, TaskStateError> {
    let Some(scheduled_for) = task.next_run_at else {
        return Ok(DueDecision::Wait);
    };
    if now < scheduled_for {
        return Ok(DueDecision::Wait);
    }

    let next_run_at = next_run_after(&task.schedule, task.timezone_offset, scheduled_for, now)?;
    let lateness = now.saturating_sub(scheduled_for);
    if lateness > MISFIRE_GRACE_SECONDS {
        return Ok(DueDecision::Skip {
            remove_task: !task.schedule.is_recurring(),
            next_run_at,
        });
    }

    Ok(DueDecision::Execute {
        execution_id: format!("{}:{scheduled_for}", task.id.as_str()),
        scheduled_for,
        next_run_at,
    })
}

/// Parse the weekday names stored by the existing Python scheduler.
pub fn parse_weekday(value: &str) -> Result<Weekday, TaskStateError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "mon" => Ok(Weekday::Mon),
        "tue" => Ok(Weekday::Tue),
        "wed" => Ok(Weekday::Wed),
        "thu" => Ok(Weekday::Thu),
        "fri" => Ok(Weekday::Fri),
        "sat" => Ok(Weekday::Sat),
        "sun" => Ok(Weekday::Sun),
        _ => Err(TaskStateError::InvalidCron),
    }
}

#[must_use]
pub const fn weekday_name(value: Weekday) -> &'static str {
    match value {
        Weekday::Mon => "mon",
        Weekday::Tue => "tue",
        Weekday::Wed => "wed",
        Weekday::Thu => "thu",
        Weekday::Fri => "fri",
        Weekday::Sat => "sat",
        Weekday::Sun => "sun",
    }
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc, Weekday};

    use super::{
        DueDecision, ScheduledTask, TaskId, TaskSchedule, TaskStateError, evaluate_due,
        initial_next_run, next_run_after, parse_weekday, weekday_name,
    };

    fn timestamp(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> i64 {
        Utc.with_ymd_and_hms(year, month, day, hour, minute, 0)
            .single()
            .map_or(0, |value| value.timestamp())
    }

    fn task(schedule: TaskSchedule, next_run_at: Option<i64>) -> ScheduledTask {
        ScheduledTask {
            id: TaskId("abc12345".to_owned()),
            chat_id: "-100123".to_owned(),
            text: "synthetic task".to_owned(),
            user_name: "synthetic-user".to_owned(),
            user_id: Some(42),
            schedule,
            timezone_offset: -3,
            locale: "es".to_owned(),
            schedule_anchor_at: None,
            next_run_at,
            last_execution_id: None,
        }
    }

    #[test]
    fn validates_task_ids() {
        assert!(TaskId::new("abc_123-X").is_ok());
        assert_eq!(TaskId::new(""), Err(TaskStateError::InvalidTaskId));
        assert_eq!(TaskId::new("has:colon"), Err(TaskStateError::InvalidTaskId));
    }

    #[test]
    fn computes_delay_and_interval_initial_runs() {
        assert_eq!(
            initial_next_run(&TaskSchedule::Once, -3, 1_000, Some(30)),
            Ok(1_030)
        );
        assert_eq!(
            initial_next_run(
                &TaskSchedule::IntervalSeconds { seconds: 600 },
                -3,
                1_000,
                None,
            ),
            Ok(1_600)
        );
        assert_eq!(
            initial_next_run(&TaskSchedule::IntervalDays { days: 2 }, -3, 1_000, None,),
            Ok(173_800)
        );
    }

    #[test]
    fn computes_daily_weekly_and_monthly_cron_in_fixed_timezone() {
        let after = timestamp(2026, 8, 30, 22, 0);
        let daily = TaskSchedule::Cron {
            hour: 20,
            minute: 30,
            weekdays: Vec::new(),
            day: None,
        };
        assert_eq!(
            initial_next_run(&daily, -3, after, None),
            Ok(timestamp(2026, 8, 30, 23, 30))
        );

        let weekly = TaskSchedule::Cron {
            hour: 9,
            minute: 0,
            weekdays: vec![Weekday::Mon],
            day: None,
        };
        assert_eq!(
            initial_next_run(&weekly, -3, after, None),
            Ok(timestamp(2026, 8, 31, 12, 0))
        );

        let monthly = TaskSchedule::Cron {
            hour: 7,
            minute: 15,
            weekdays: Vec::new(),
            day: Some(1),
        };
        assert_eq!(
            initial_next_run(&monthly, -3, after, None),
            Ok(timestamp(2026, 9, 1, 10, 15))
        );
    }

    #[test]
    fn coalesces_interval_occurrences_and_keeps_alignment() {
        let schedule = TaskSchedule::IntervalSeconds { seconds: 600 };
        assert_eq!(next_run_after(&schedule, -3, 1_000, 2_450), Ok(Some(2_800)));
    }

    #[test]
    fn waits_executes_and_skips_using_existing_grace_policy() {
        let scheduled = 10_000;
        let recurring = task(
            TaskSchedule::IntervalSeconds { seconds: 600 },
            Some(scheduled),
        );
        assert_eq!(evaluate_due(&recurring, 9_999), Ok(DueDecision::Wait));
        assert_eq!(
            evaluate_due(&recurring, 10_300),
            Ok(DueDecision::Execute {
                execution_id: "abc12345:10000".to_owned(),
                scheduled_for: 10_000,
                next_run_at: Some(10_600),
            })
        );
        assert_eq!(
            evaluate_due(&recurring, 11_201),
            Ok(DueDecision::Skip {
                remove_task: false,
                next_run_at: Some(11_800),
            })
        );

        let one_shot = task(TaskSchedule::Once, Some(scheduled));
        assert_eq!(
            evaluate_due(&one_shot, 10_301),
            Ok(DueDecision::Skip {
                remove_task: true,
                next_run_at: None,
            })
        );
    }

    #[test]
    fn rejects_invalid_schedule_boundaries() {
        assert_eq!(
            initial_next_run(
                &TaskSchedule::IntervalSeconds { seconds: 0 },
                -3,
                1_000,
                None,
            ),
            Err(TaskStateError::InvalidInterval)
        );
        assert_eq!(
            initial_next_run(
                &TaskSchedule::Cron {
                    hour: 25,
                    minute: 0,
                    weekdays: Vec::new(),
                    day: None,
                },
                -3,
                1_000,
                None,
            ),
            Err(TaskStateError::InvalidCron)
        );
        assert_eq!(
            initial_next_run(
                &TaskSchedule::Cron {
                    hour: 0,
                    minute: 0,
                    weekdays: Vec::new(),
                    day: None,
                },
                15,
                1_000,
                None,
            ),
            Err(TaskStateError::InvalidTimezone)
        );
    }

    #[test]
    fn weekday_storage_names_round_trip() {
        for (name, weekday) in [
            ("mon", Weekday::Mon),
            ("tue", Weekday::Tue),
            ("wed", Weekday::Wed),
            ("thu", Weekday::Thu),
            ("fri", Weekday::Fri),
            ("sat", Weekday::Sat),
            ("sun", Weekday::Sun),
        ] {
            assert_eq!(parse_weekday(name), Ok(weekday));
            assert_eq!(weekday_name(weekday), name);
        }
        assert_eq!(parse_weekday("bad"), Err(TaskStateError::InvalidCron));
    }
}
