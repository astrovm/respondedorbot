//! Versioned, rollback-compatible scheduled-task JSON records.

use std::collections::BTreeMap;

use bot_core::scheduled_tasks::{
    ScheduledTask, TASK_SCHEMA_VERSION, TaskId, TaskSchedule, TaskStateError, parse_weekday,
    weekday_name,
};
use chrono::{SecondsFormat, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub struct TaskRecordDocument {
    pub task: ScheduledTask,
    /// Original one-shot date retained for rollback readers.
    pub legacy_run_date: Option<String>,
    /// Fields from newer compatible writers that this adapter does not own.
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Error)]
pub enum TaskRecordError {
    #[error("scheduled-task JSON is malformed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("scheduled-task schema version {0} is unsupported")]
    UnsupportedVersion(u64),
    #[error("scheduled-task field `{0}` is missing or malformed")]
    InvalidField(&'static str),
    #[error("scheduled-task trigger is missing or malformed")]
    InvalidTrigger,
    #[error("scheduled-task timestamp `{0}` is malformed")]
    InvalidTimestamp(&'static str),
    #[error(transparent)]
    State(#[from] TaskStateError),
}

#[derive(Debug, Deserialize)]
struct RawTaskRecord {
    #[serde(default)]
    schema_version: Option<u64>,
    id: Value,
    chat_id: Value,
    text: Value,
    #[serde(default)]
    user_name: Value,
    #[serde(default)]
    user_id: Option<i64>,
    #[serde(default)]
    interval_seconds: Option<i64>,
    #[serde(default)]
    run_date: Option<String>,
    #[serde(default)]
    trigger_config: Option<Value>,
    #[serde(default)]
    timezone_offset: Option<i32>,
    #[serde(default)]
    locale: Option<String>,
    #[serde(default)]
    schedule_anchor_at: Option<String>,
    #[serde(default)]
    next_run_at: Option<String>,
    #[serde(default)]
    last_execution_id: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Serialize)]
struct CanonicalTaskRecord<'a> {
    schema_version: u8,
    id: &'a str,
    chat_id: &'a str,
    text: &'a str,
    user_name: &'a str,
    user_id: Option<i64>,
    interval_seconds: Option<i64>,
    run_date: Option<&'a str>,
    trigger_config: Option<Value>,
    timezone_offset: i32,
    locale: &'a str,
    schedule_anchor_at: Option<String>,
    next_run_at: Option<String>,
    last_execution_id: Option<&'a str>,
    #[serde(flatten)]
    extra: &'a BTreeMap<String, Value>,
}

fn required_string(value: &Value, field: &'static str) -> Result<String, TaskRecordError> {
    let value = match value {
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        _ => return Err(TaskRecordError::InvalidField(field)),
    };
    if value.is_empty() {
        return Err(TaskRecordError::InvalidField(field));
    }
    Ok(value)
}

fn optional_string(value: &Value, field: &'static str) -> Result<String, TaskRecordError> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(value) => Ok(value.clone()),
        Value::Number(value) => Ok(value.to_string()),
        _ => Err(TaskRecordError::InvalidField(field)),
    }
}

fn timestamp(value: Option<&str>, field: &'static str) -> Result<Option<i64>, TaskRecordError> {
    value
        .map(|value| {
            chrono::DateTime::parse_from_rfc3339(value)
                .map(|value| value.timestamp())
                .map_err(|_| TaskRecordError::InvalidTimestamp(field))
        })
        .transpose()
}

fn timestamp_text(value: Option<i64>) -> Result<Option<String>, TaskRecordError> {
    value
        .map(|value| {
            Utc.timestamp_opt(value, 0)
                .single()
                .map(|value| value.to_rfc3339_opts(SecondsFormat::Secs, true))
                .ok_or(TaskRecordError::InvalidTimestamp("timestamp"))
        })
        .transpose()
}

fn integer_field(object: &Map<String, Value>, field: &'static str) -> Result<i64, TaskRecordError> {
    object
        .get(field)
        .and_then(Value::as_i64)
        .ok_or(TaskRecordError::InvalidTrigger)
}

fn parse_trigger(raw: &RawTaskRecord) -> Result<TaskSchedule, TaskRecordError> {
    if let Some(seconds) = raw.interval_seconds {
        if !(300..=604_800).contains(&seconds) {
            return Err(TaskRecordError::InvalidTrigger);
        }
        return Ok(TaskSchedule::IntervalSeconds { seconds });
    }

    if let Some(Value::Object(config)) = raw.trigger_config.as_ref() {
        match config.get("type").and_then(Value::as_str) {
            Some("interval") => {
                let days = integer_field(config, "days")?;
                if !(1..=90).contains(&days) {
                    return Err(TaskRecordError::InvalidTrigger);
                }
                return Ok(TaskSchedule::IntervalDays { days });
            }
            Some("cron") => {
                let hour = u32::try_from(integer_field(config, "hour")?)
                    .map_err(|_| TaskRecordError::InvalidTrigger)?;
                let minute = u32::try_from(integer_field(config, "minute")?)
                    .map_err(|_| TaskRecordError::InvalidTrigger)?;
                if hour > 23 || minute > 59 {
                    return Err(TaskRecordError::InvalidTrigger);
                }
                let weekdays = config
                    .get("day_of_week")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .split(',')
                    .filter(|part| !part.trim().is_empty())
                    .map(parse_weekday)
                    .collect::<Result<Vec<_>, _>>()?;
                let day = config
                    .get("day")
                    .filter(|value| !value.is_null())
                    .map(|value| {
                        value
                            .as_u64()
                            .and_then(|value| u32::try_from(value).ok())
                            .filter(|value| (1..=31).contains(value))
                            .ok_or(TaskRecordError::InvalidTrigger)
                    })
                    .transpose()?;
                return Ok(TaskSchedule::Cron {
                    hour,
                    minute,
                    weekdays,
                    day,
                });
            }
            _ => return Err(TaskRecordError::InvalidTrigger),
        }
    }

    if raw.run_date.is_some() || raw.next_run_at.is_some() {
        return Ok(TaskSchedule::Once);
    }
    Err(TaskRecordError::InvalidTrigger)
}

fn trigger_fields(schedule: &TaskSchedule) -> (Option<i64>, Option<Value>) {
    match schedule {
        TaskSchedule::Once => (None, None),
        TaskSchedule::IntervalSeconds { seconds } => (Some(*seconds), None),
        TaskSchedule::IntervalDays { days } => {
            (None, Some(json!({"type":"interval", "days":days})))
        }
        TaskSchedule::Cron {
            hour,
            minute,
            weekdays,
            day,
        } => {
            let mut config = Map::from_iter([
                ("type".to_owned(), Value::String("cron".to_owned())),
                ("hour".to_owned(), Value::from(*hour)),
                ("minute".to_owned(), Value::from(*minute)),
            ]);
            if !weekdays.is_empty() {
                config.insert(
                    "day_of_week".to_owned(),
                    Value::String(
                        weekdays
                            .iter()
                            .map(|value| weekday_name(*value))
                            .collect::<Vec<_>>()
                            .join(","),
                    ),
                );
            }
            if let Some(day) = day {
                config.insert("day".to_owned(), Value::from(*day));
            }
            (None, Some(Value::Object(config)))
        }
    }
}

/// Decode both legacy unversioned and canonical version 1 records.
pub fn decode_task_record(payload: &str) -> Result<TaskRecordDocument, TaskRecordError> {
    let raw = serde_json::from_str::<RawTaskRecord>(payload)?;
    match raw.schema_version {
        None | Some(0) => {}
        Some(version) if version == u64::from(TASK_SCHEMA_VERSION) => {}
        Some(version) => return Err(TaskRecordError::UnsupportedVersion(version)),
    }

    let id = TaskId::new(required_string(&raw.id, "id")?)?;
    let chat_id = required_string(&raw.chat_id, "chat_id")?;
    let text = required_string(&raw.text, "text")?;
    let user_name = optional_string(&raw.user_name, "user_name")?;
    let schedule = parse_trigger(&raw)?;
    let timezone_offset = raw.timezone_offset.unwrap_or(-3);
    if !(-12..=14).contains(&timezone_offset) {
        return Err(TaskRecordError::InvalidField("timezone_offset"));
    }
    let locale = match raw.locale.as_deref() {
        Some("en") => "en",
        _ => "es",
    }
    .to_owned();
    let run_date = timestamp(raw.run_date.as_deref(), "run_date")?;
    let next_run_at = timestamp(raw.next_run_at.as_deref(), "next_run_at")?.or(run_date);

    Ok(TaskRecordDocument {
        task: ScheduledTask {
            id,
            chat_id,
            text,
            user_name,
            user_id: raw.user_id,
            schedule,
            timezone_offset,
            locale,
            schedule_anchor_at: timestamp(raw.schedule_anchor_at.as_deref(), "schedule_anchor_at")?,
            next_run_at,
            last_execution_id: raw.last_execution_id,
        },
        legacy_run_date: raw.run_date,
        extra: raw.extra,
    })
}

/// Encode the additive version 1 payload while retaining unknown compatible
/// fields and legacy trigger representation.
pub fn encode_task_record(document: &TaskRecordDocument) -> Result<String, TaskRecordError> {
    let (interval_seconds, trigger_config) = trigger_fields(&document.task.schedule);
    let run_date = if matches!(document.task.schedule, TaskSchedule::Once) {
        document.legacy_run_date.as_deref()
    } else {
        None
    };
    Ok(serde_json::to_string(&CanonicalTaskRecord {
        schema_version: TASK_SCHEMA_VERSION,
        id: document.task.id.as_str(),
        chat_id: &document.task.chat_id,
        text: &document.task.text,
        user_name: &document.task.user_name,
        user_id: document.task.user_id,
        interval_seconds,
        run_date,
        trigger_config,
        timezone_offset: document.task.timezone_offset,
        locale: &document.task.locale,
        schedule_anchor_at: timestamp_text(document.task.schedule_anchor_at)?,
        next_run_at: timestamp_text(document.task.next_run_at)?,
        last_execution_id: document.task.last_execution_id.as_deref(),
        extra: &document.extra,
    })?)
}

/// Upgrade a readable record to canonical version 1 without changing its
/// schedule. A recurring legacy record may still need its APScheduler next run
/// injected before it is safe for cutover.
pub fn normalize_task_record(payload: &str) -> Result<String, TaskRecordError> {
    encode_task_record(&decode_task_record(payload)?)
}

#[cfg(test)]
mod tests {
    use bot_core::scheduled_tasks::TaskSchedule;
    use chrono::Weekday;
    use serde::Deserialize;
    use serde_json::{Value, json};

    use super::{
        TASK_SCHEMA_VERSION, TaskRecordError, decode_task_record, encode_task_record,
        normalize_task_record,
    };

    #[test]
    fn upgrades_a_legacy_one_shot_and_uses_run_date_as_next_run() -> Result<(), TaskRecordError> {
        let normalized = normalize_task_record(
            r#"{"id":"abc12345","chat_id":"-100123","text":"synthetic task","user_name":"user","user_id":42,"interval_seconds":null,"run_date":"2026-08-30T12:00:00+00:00","trigger_config":null}"#,
        )?;
        let value: Value = serde_json::from_str(&normalized)?;
        assert_eq!(value["schema_version"], TASK_SCHEMA_VERSION);
        assert_eq!(value["next_run_at"], "2026-08-30T12:00:00Z");
        assert_eq!(value["run_date"], "2026-08-30T12:00:00+00:00");
        Ok(())
    }

    #[test]
    fn round_trips_canonical_cron_and_preserves_unknown_fields() -> Result<(), TaskRecordError> {
        let payload = json!({
            "schema_version": 1,
            "id": "abc12345",
            "chat_id": "-100123",
            "text": "synthetic task",
            "user_name": "user",
            "user_id": 42,
            "interval_seconds": null,
            "run_date": null,
            "trigger_config": {
                "type": "cron", "hour": 9, "minute": 5, "day_of_week": "mon,wed"
            },
            "timezone_offset": -3,
            "locale": "en",
            "schedule_anchor_at": "2026-08-30T10:00:00Z",
            "next_run_at": "2026-08-31T12:05:00Z",
            "last_execution_id": "abc12345:1",
            "future_field": {"kept": true}
        })
        .to_string();
        let document = decode_task_record(&payload)?;
        assert_eq!(
            document.task.schedule,
            TaskSchedule::Cron {
                hour: 9,
                minute: 5,
                weekdays: vec![Weekday::Mon, Weekday::Wed],
                day: None,
            }
        );
        let encoded: Value = serde_json::from_str(&encode_task_record(&document)?)?;
        assert_eq!(encoded["future_field"], json!({"kept":true}));
        assert_eq!(encoded["last_execution_id"], "abc12345:1");
        Ok(())
    }

    #[test]
    fn accepts_numeric_legacy_chat_ids_and_defaults_locale_and_timezone()
    -> Result<(), TaskRecordError> {
        let document = decode_task_record(
            r#"{"id":"abc12345","chat_id":123,"text":"synthetic task","user_name":null,"interval_seconds":600,"run_date":null,"trigger_config":null}"#,
        )?;
        assert_eq!(document.task.chat_id, "123");
        assert_eq!(document.task.timezone_offset, -3);
        assert_eq!(document.task.locale, "es");
        assert_eq!(
            document.task.schedule,
            TaskSchedule::IntervalSeconds { seconds: 600 }
        );
        assert_eq!(document.task.next_run_at, None);
        Ok(())
    }

    #[test]
    fn rejects_unsupported_versions_and_malformed_records() {
        assert!(matches!(
            decode_task_record(
                r#"{"schema_version":2,"id":"abc12345","chat_id":"1","text":"x","run_date":"2026-08-30T12:00:00Z"}"#
            ),
            Err(TaskRecordError::UnsupportedVersion(2))
        ));
        assert!(matches!(
            decode_task_record(
                r#"{"id":"bad:id","chat_id":"1","text":"x","run_date":"2026-08-30T12:00:00Z"}"#
            ),
            Err(TaskRecordError::State(_))
        ));
        assert!(matches!(
            decode_task_record(
                r#"{"id":"abc12345","chat_id":"1","text":"x","trigger_config":{"type":"cron","hour":99,"minute":0}}"#
            ),
            Err(TaskRecordError::InvalidTrigger)
        ));
        assert!(matches!(
            decode_task_record(
                r#"{"id":"abc12345","chat_id":"1","text":"x","run_date":"not-a-date"}"#
            ),
            Err(TaskRecordError::InvalidTimestamp("run_date"))
        ));
    }

    #[test]
    fn contract_examples_normalize_to_their_expected_values()
    -> Result<(), Box<dyn std::error::Error>> {
        #[derive(Deserialize)]
        struct Contract {
            cases: Vec<Case>,
        }
        #[derive(Deserialize)]
        struct Case {
            input: Value,
            expected: Value,
        }

        let contract: Contract =
            serde_json::from_str(include_str!("../../../contracts/task_records.json"))?;
        for case in contract.cases {
            let normalized = normalize_task_record(&case.input.to_string())?;
            let value: Value = serde_json::from_str(&normalized)?;
            for (key, expected) in case.expected.as_object().into_iter().flatten() {
                assert_eq!(&value[key], expected);
            }
        }
        Ok(())
    }
}
