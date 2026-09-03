//! One-time, idempotent migration of persisted Python-era records.

use std::collections::HashMap;

use bot_core::scheduled_tasks::{TaskSchedule, initial_next_run};
use postgres::Client;
use redis::Commands;
use serde::Serialize;
use serde_json::Value;
use thiserror::Error;

use crate::postgres_connection::postgres_tls_connector;
use crate::redis_connection::{RedisEndpoint, connect};
use crate::task_record::{TaskRecordDocument, decode_task_record, encode_task_record};

const CURRENT_SCHEMA_VERSION: u64 = 1;
const TASK_RECORD_TTL_SECONDS: i64 = 86_400 * 3_650;
const CHAT_CONFIG_MIGRATION_LOCK: i64 = 48_610_007;
const UPSERT_TASK_IF_UNCHANGED: &str = r#"
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('SETEX', KEYS[1], ARGV[2], ARGV[3])
redis.call('ZADD', KEYS[2], ARGV[4], ARGV[5])
redis.call('EXPIRE', KEYS[2], ARGV[2])
redis.call('SETEX', KEYS[3], ARGV[2], '1')
redis.call('ZADD', KEYS[4], ARGV[4], ARGV[5])
return 1
"#;
const REPLACE_LIST_ITEM_IF_UNCHANGED: &str = r#"
if redis.call('LINDEX', KEYS[1], ARGV[1]) ~= ARGV[2] then return 0 end
redis.call('LSET', KEYS[1], ARGV[1], ARGV[3])
return 1
"#;
const REPLACE_HASH_ITEM_IF_UNCHANGED: &str = r#"
if redis.call('HGET', KEYS[1], ARGV[1]) ~= ARGV[2] then return 0 end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[3])
return 1
"#;
const REPLACE_STRING_IF_UNCHANGED: &str = r#"
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('SET', KEYS[1], ARGV[2], 'KEEPTTL')
return 1
"#;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MigrationMode {
    DryRun,
    Apply,
}

impl MigrationMode {
    const fn applies(self) -> bool {
        matches!(self, Self::Apply)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct RecordMigrationReport {
    pub scanned: usize,
    pub current: usize,
    pub candidates: usize,
    pub migrated: usize,
    pub malformed: usize,
    pub unsupported: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct RedisMigrationReport {
    pub history: RecordMigrationReport,
    pub chat_members: RecordMigrationReport,
    pub bot_metadata: RecordMigrationReport,
    pub tasks_scanned: usize,
    pub tasks_upgraded: usize,
    pub task_indexes_rebuilt: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct PostgresMigrationReport {
    pub obsolete_chat_config_fields: u64,
    pub removed_chat_config_fields: u64,
}

#[derive(Debug, Error)]
pub enum LegacyMigrationError {
    #[error("Redis migration failed: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("PostgreSQL migration failed: {0}")]
    Postgres(#[from] postgres::Error),
    #[error("could not initialize PostgreSQL TLS: {0}")]
    Tls(#[from] native_tls::Error),
    #[error("{family} contains malformed JSON")]
    MalformedJson { family: &'static str },
    #[error("{family} contains unsupported schema version {version}")]
    UnsupportedVersion { family: &'static str, version: u64 },
    #[error("scheduled-task migration failed: {0}")]
    TaskRecord(#[from] crate::task_record::TaskRecordError),
    #[error("scheduled-task recurrence migration failed: {0}")]
    TaskState(#[from] bot_core::scheduled_tasks::TaskStateError),
    #[error("stored data changed while migrating {family}; retry the idempotent migration")]
    ConcurrentChange { family: &'static str },
}

#[derive(Clone, Debug)]
struct Change {
    key: String,
    field: ChangeField,
    previous: String,
    replacement: String,
}

#[derive(Clone, Debug)]
enum ChangeField {
    List(usize),
    Hash(String),
    String,
}

#[derive(Clone, Debug)]
struct TaskChange {
    previous: String,
    document: TaskRecordDocument,
    replacement: String,
}

fn versioned_payload(
    family: &'static str,
    payload: &str,
) -> Result<Option<String>, LegacyMigrationError> {
    let mut value = serde_json::from_str::<Value>(payload)
        .map_err(|_| LegacyMigrationError::MalformedJson { family })?;
    let object = value
        .as_object_mut()
        .ok_or(LegacyMigrationError::MalformedJson { family })?;
    match object.get("schema_version") {
        None | Some(Value::Null) => {}
        Some(Value::Number(version)) if version.as_u64() == Some(0) => {}
        Some(Value::Number(version)) if version.as_u64() == Some(CURRENT_SCHEMA_VERSION) => {
            return Ok(None);
        }
        Some(Value::Number(version)) => {
            return Err(LegacyMigrationError::UnsupportedVersion {
                family,
                version: version.as_u64().unwrap_or(u64::MAX),
            });
        }
        Some(_) => return Err(LegacyMigrationError::MalformedJson { family }),
    }
    object.insert(
        "schema_version".to_owned(),
        Value::from(CURRENT_SCHEMA_VERSION),
    );
    serde_json::to_string(&value)
        .map(Some)
        .map_err(|_| LegacyMigrationError::MalformedJson { family })
}

fn scan_keys(
    connection: &mut redis::Connection,
    pattern: &str,
) -> Result<Vec<String>, redis::RedisError> {
    let mut cursor = 0_u64;
    let mut keys = Vec::new();
    loop {
        let (next_cursor, page): (u64, Vec<String>) = redis::cmd("SCAN")
            .arg(cursor)
            .arg("MATCH")
            .arg(pattern)
            .arg("COUNT")
            .arg(100)
            .query(connection)?;
        keys.extend(page);
        cursor = next_cursor;
        if cursor == 0 {
            keys.sort_unstable();
            return Ok(keys);
        }
    }
}

fn list_changes(
    connection: &mut redis::Connection,
    pattern: &str,
    family: &'static str,
) -> Result<(RecordMigrationReport, Vec<Change>), LegacyMigrationError> {
    let mut report = RecordMigrationReport::default();
    let mut changes = Vec::new();
    for key in scan_keys(connection, pattern)? {
        let values: Vec<String> = connection.lrange(&key, 0, -1)?;
        for (index, previous) in values.into_iter().enumerate() {
            report.scanned += 1;
            match versioned_payload(family, &previous) {
                Ok(Some(replacement)) => {
                    report.candidates += 1;
                    changes.push(Change {
                        key: key.clone(),
                        field: ChangeField::List(index),
                        previous,
                        replacement,
                    });
                }
                Ok(None) => report.current += 1,
                Err(LegacyMigrationError::MalformedJson { .. }) => report.malformed += 1,
                Err(LegacyMigrationError::UnsupportedVersion { .. }) => report.unsupported += 1,
                Err(error) => return Err(error),
            }
        }
    }
    Ok((report, changes))
}

fn hash_changes(
    connection: &mut redis::Connection,
    pattern: &str,
    family: &'static str,
) -> Result<(RecordMigrationReport, Vec<Change>), LegacyMigrationError> {
    let mut report = RecordMigrationReport::default();
    let mut changes = Vec::new();
    for key in scan_keys(connection, pattern)? {
        let values: HashMap<String, String> = connection.hgetall(&key)?;
        for (field, previous) in values {
            report.scanned += 1;
            match versioned_payload(family, &previous) {
                Ok(Some(replacement)) => {
                    report.candidates += 1;
                    changes.push(Change {
                        key: key.clone(),
                        field: ChangeField::Hash(field),
                        previous,
                        replacement,
                    });
                }
                Ok(None) => report.current += 1,
                Err(LegacyMigrationError::MalformedJson { .. }) => report.malformed += 1,
                Err(LegacyMigrationError::UnsupportedVersion { .. }) => report.unsupported += 1,
                Err(error) => return Err(error),
            }
        }
    }
    Ok((report, changes))
}

fn string_changes(
    connection: &mut redis::Connection,
    pattern: &str,
    family: &'static str,
) -> Result<(RecordMigrationReport, Vec<Change>), LegacyMigrationError> {
    let mut report = RecordMigrationReport::default();
    let mut changes = Vec::new();
    for key in scan_keys(connection, pattern)? {
        let Some(previous): Option<String> = connection.get(&key)? else {
            continue;
        };
        report.scanned += 1;
        match versioned_payload(family, &previous) {
            Ok(Some(replacement)) => {
                report.candidates += 1;
                changes.push(Change {
                    key,
                    field: ChangeField::String,
                    previous,
                    replacement,
                });
            }
            Ok(None) => report.current += 1,
            Err(LegacyMigrationError::MalformedJson { .. }) => report.malformed += 1,
            Err(LegacyMigrationError::UnsupportedVersion { .. }) => report.unsupported += 1,
            Err(error) => return Err(error),
        }
    }
    Ok((report, changes))
}

fn apply_changes(
    connection: &mut redis::Connection,
    family: &'static str,
    changes: &[Change],
) -> Result<(), LegacyMigrationError> {
    for change in changes {
        let changed = match &change.field {
            ChangeField::List(index) => redis::cmd("EVAL")
                .arg(REPLACE_LIST_ITEM_IF_UNCHANGED)
                .arg(1)
                .arg(&change.key)
                .arg(*index)
                .arg(&change.previous)
                .arg(&change.replacement)
                .query::<i64>(connection)?,
            ChangeField::Hash(field) => redis::cmd("EVAL")
                .arg(REPLACE_HASH_ITEM_IF_UNCHANGED)
                .arg(1)
                .arg(&change.key)
                .arg(field)
                .arg(&change.previous)
                .arg(&change.replacement)
                .query::<i64>(connection)?,
            ChangeField::String => redis::cmd("EVAL")
                .arg(REPLACE_STRING_IF_UNCHANGED)
                .arg(1)
                .arg(&change.key)
                .arg(&change.previous)
                .arg(&change.replacement)
                .query::<i64>(connection)?,
        };
        if changed != 1 {
            return Err(LegacyMigrationError::ConcurrentChange { family });
        }
    }
    Ok(())
}

fn task_change(payload: String, now: i64) -> Result<TaskChange, LegacyMigrationError> {
    let mut document = decode_task_record(&payload)?;
    if document.task.next_run_at.is_none() {
        let delay = matches!(document.task.schedule, TaskSchedule::Once).then_some(1);
        document.task.next_run_at = Some(initial_next_run(
            &document.task.schedule,
            document.task.timezone_offset,
            now,
            delay,
        )?);
    }
    if document.task.schedule_anchor_at.is_none() && document.task.schedule.is_recurring() {
        document.task.schedule_anchor_at = Some(now);
    }
    let replacement = encode_task_record(&document)?;
    Ok(TaskChange {
        previous: payload,
        document,
        replacement,
    })
}

fn apply_task_change(
    connection: &mut redis::Connection,
    change: &TaskChange,
) -> Result<(), LegacyMigrationError> {
    let task = &change.document.task;
    let next_run_at = task
        .next_run_at
        .ok_or(LegacyMigrationError::ConcurrentChange {
            family: "scheduled tasks",
        })?;
    let data_key = format!("task:data:{}", task.id.as_str());
    let chat_key = format!("task:chat:{}", task.chat_id);
    let marker_key = format!("{chat_key}:indexed");
    let changed: i64 = redis::cmd("EVAL")
        .arg(UPSERT_TASK_IF_UNCHANGED)
        .arg(4)
        .arg(data_key)
        .arg(chat_key)
        .arg(marker_key)
        .arg("task:due")
        .arg(&change.previous)
        .arg(TASK_RECORD_TTL_SECONDS)
        .arg(&change.replacement)
        .arg(next_run_at)
        .arg(task.id.as_str())
        .query(connection)?;
    if changed == 1 {
        Ok(())
    } else {
        Err(LegacyMigrationError::ConcurrentChange {
            family: "scheduled tasks",
        })
    }
}

pub fn migrate_redis(
    endpoint: &RedisEndpoint,
    mode: MigrationMode,
    now: i64,
) -> Result<RedisMigrationReport, LegacyMigrationError> {
    let mut connection = connect(endpoint)?;
    let (mut history, history_changes) =
        list_changes(&mut connection, "chat_history:*", "chat history")?;
    let (mut chat_members, member_changes) =
        hash_changes(&mut connection, "chat_members:*", "chat members")?;
    let (mut bot_metadata, metadata_changes) = string_changes(
        &mut connection,
        "bot_message_meta:*",
        "bot message metadata",
    )?;

    let mut task_changes = Vec::new();
    for key in scan_keys(&mut connection, "task:data:*")? {
        if let Some(payload) = connection.get::<_, Option<String>>(key)? {
            task_changes.push(task_change(payload, now)?);
        }
    }
    let tasks_upgraded = task_changes
        .iter()
        .filter(|change| change.previous != change.replacement)
        .count();

    if mode.applies() {
        apply_changes(&mut connection, "chat history", &history_changes)?;
        apply_changes(&mut connection, "chat members", &member_changes)?;
        apply_changes(&mut connection, "bot message metadata", &metadata_changes)?;
        for change in &task_changes {
            apply_task_change(&mut connection, change)?;
        }
        history.migrated = history_changes.len();
        chat_members.migrated = member_changes.len();
        bot_metadata.migrated = metadata_changes.len();
    }

    Ok(RedisMigrationReport {
        history,
        chat_members,
        bot_metadata,
        tasks_scanned: task_changes.len(),
        tasks_upgraded,
        task_indexes_rebuilt: if mode.applies() {
            task_changes.len()
        } else {
            0
        },
    })
}

pub fn migrate_postgres(
    database_url: &str,
    mode: MigrationMode,
) -> Result<PostgresMigrationReport, LegacyMigrationError> {
    let mut client = Client::connect(database_url, postgres_tls_connector(database_url)?)?;
    let mut transaction = client.transaction()?;
    transaction.query_one(
        "SELECT pg_advisory_xact_lock($1)",
        &[&CHAT_CONFIG_MIGRATION_LOCK],
    )?;
    let table_exists = transaction
        .query_one("SELECT to_regclass('public.chat_configs')::text", &[])?
        .get::<_, Option<String>>(0)
        .is_some();
    if !table_exists {
        transaction.commit()?;
        return Ok(PostgresMigrationReport::default());
    }
    let candidates = transaction
        .query_one(
            "SELECT COUNT(*) FROM chat_configs WHERE config ? 'world_cup_goal_alerts'",
            &[],
        )?
        .get::<_, i64>(0)
        .max(0) as u64;
    let removed = if mode.applies() {
        transaction.execute(
            "UPDATE chat_configs SET config = config - 'world_cup_goal_alerts', \
             updated_at = NOW() WHERE config ? 'world_cup_goal_alerts'",
            &[],
        )?
    } else {
        0
    };
    transaction.commit()?;
    Ok(PostgresMigrationReport {
        obsolete_chat_config_fields: candidates,
        removed_chat_config_fields: removed,
    })
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::io::Write;
    use std::net::TcpListener;
    use std::thread;

    use chrono::{TimeZone, Utc};
    use postgres::Client;
    use serde_json::{Value, json};

    use super::{
        Change, ChangeField, LegacyMigrationError, MigrationMode, apply_changes, apply_task_change,
        task_change, versioned_payload,
    };
    use crate::chat_config::ChatConfigRepository;
    use crate::legacy_migration::{migrate_postgres, migrate_redis};
    use crate::postgres_connection::postgres_tls_connector;
    use crate::redis_connection::{RedisEndpoint, connect, test_support::read_command};

    fn bulk(value: &str) -> String {
        format!("${}\r\n{}\r\n", value.len(), value)
    }

    fn array(values: &[&str]) -> String {
        let mut response = format!("*{}\r\n", values.len());
        for value in values {
            response.push_str(&bulk(value));
        }
        response
    }

    fn scan(keys: &[&str]) -> String {
        format!("*2\r\n$1\r\n0\r\n{}", array(keys))
    }

    fn scan_page(cursor: &str, keys: &[&str]) -> String {
        format!("*2\r\n{}{}", bulk(cursor), array(keys))
    }

    fn migration_server(listener: TcpListener) -> thread::JoinHandle<Result<(), String>> {
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().map_err(|error| error.to_string())?;
            let history = r#"{"id":"1","text":"synthetic","timestamp":1,"role":"user"}"#;
            let member = r#"{"first_name":"Synthetic","username":"tester","last_seen":1}"#;
            let metadata = r#"{"type":"ai"}"#;
            let task = json!({
                "id":"synthetic-task",
                "chat_id":"synthetic-chat",
                "text":"synthetic reminder",
                "user_name":"synthetic-user",
                "user_id":42,
                "interval_seconds":null,
                "run_date":null,
                "trigger_config":{"type":"cron","hour":20,"minute":30},
                "timezone_offset":-3
            })
            .to_string();
            let mut evaluations = 0;
            loop {
                let command = match read_command(&mut stream) {
                    Ok(command) => command,
                    Err(_) if evaluations == 4 => return Ok(()),
                    Err(error) => return Err(error.to_string()),
                };
                let response = match command.first().map(String::as_str) {
                    Some("SCAN") => match command.get(3).map(String::as_str) {
                        Some("chat_history:*") => scan(&["chat_history:synthetic"]),
                        Some("chat_members:*") => scan(&["chat_members:synthetic"]),
                        Some("bot_message_meta:*") => scan(&["bot_message_meta:synthetic:1"]),
                        Some("task:data:*") => scan(&["task:data:synthetic-task"]),
                        _ => return Err("unexpected scan pattern".to_owned()),
                    },
                    Some("LRANGE") => array(&[history]),
                    Some("HGETALL") => array(&["42", member]),
                    Some("GET") if command.get(1).is_some_and(|key| key.starts_with("bot_")) => {
                        bulk(metadata)
                    }
                    Some("GET") => bulk(&task),
                    Some("EVAL") => {
                        let replacement = command.last().map(String::as_str).unwrap_or_default();
                        if evaluations < 3 && !replacement.contains("schema_version") {
                            return Err("record replacement was not versioned".to_owned());
                        }
                        evaluations += 1;
                        ":1\r\n".to_owned()
                    }
                    _ => return Err("unexpected migration command".to_owned()),
                };
                stream
                    .write_all(response.as_bytes())
                    .map_err(|error| error.to_string())?;
                if evaluations == 4 {
                    return Ok(());
                }
            }
        })
    }

    fn inspection_server(listener: TcpListener) -> thread::JoinHandle<Result<(), String>> {
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().map_err(|error| error.to_string())?;
            let legacy = r#"{"value":"legacy"}"#;
            let current = r#"{"schema_version":1,"value":"current"}"#;
            let malformed = "not-json";
            let unsupported = r#"{"schema_version":2,"value":"future"}"#;
            let mut history_scan = 0;
            let mut commands = 0;
            loop {
                let command = read_command(&mut stream).map_err(|error| error.to_string())?;
                commands += 1;
                let response = match command.first().map(String::as_str) {
                    Some("SCAN") => match command.get(3).map(String::as_str) {
                        Some("chat_history:*") if history_scan == 0 => {
                            history_scan += 1;
                            scan_page("1", &["chat_history:synthetic-a"])
                        }
                        Some("chat_history:*") => scan(&["chat_history:synthetic-b"]),
                        Some("chat_members:*") => scan(&["chat_members:synthetic"]),
                        Some("bot_message_meta:*") => scan(&[
                            "bot_message_meta:synthetic:current",
                            "bot_message_meta:synthetic:malformed",
                            "bot_message_meta:synthetic:missing",
                            "bot_message_meta:synthetic:unsupported",
                        ]),
                        Some("task:data:*") => scan(&["task:data:synthetic-missing"]),
                        _ => return Err("unexpected scan pattern".to_owned()),
                    },
                    Some("LRANGE")
                        if command
                            .get(1)
                            .is_some_and(|key| key.ends_with("synthetic-a")) =>
                    {
                        array(&[legacy, current, malformed, unsupported])
                    }
                    Some("LRANGE") => array(&[]),
                    Some("HGETALL") => array(&[
                        "legacy",
                        legacy,
                        "current",
                        current,
                        "malformed",
                        malformed,
                        "unsupported",
                        unsupported,
                    ]),
                    Some("GET") => match command.get(1).map(String::as_str) {
                        Some(key) if key.ends_with("current") => bulk(current),
                        Some(key) if key.ends_with("malformed") => bulk(malformed),
                        Some(key) if key.ends_with("unsupported") => bulk(unsupported),
                        Some(_) => "$-1\r\n".to_owned(),
                        None => return Err("missing GET key".to_owned()),
                    },
                    _ => return Err("unexpected inspection command".to_owned()),
                };
                stream
                    .write_all(response.as_bytes())
                    .map_err(|error| error.to_string())?;
                if commands == 13 {
                    return Ok(());
                }
            }
        })
    }

    fn integer_server(listener: TcpListener, value: i64) -> thread::JoinHandle<Result<(), String>> {
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().map_err(|error| error.to_string())?;
            read_command(&mut stream).map_err(|error| error.to_string())?;
            stream
                .write_all(format!(":{value}\r\n").as_bytes())
                .map_err(|error| error.to_string())
        })
    }

    #[test]
    fn record_versioning_is_idempotent_and_rejects_invalid_inputs() {
        let migrated = versioned_payload("synthetic records", r#"{"value":1}"#);
        assert_eq!(
            migrated
                .ok()
                .flatten()
                .and_then(|value| serde_json::from_str::<serde_json::Value>(&value).ok())
                .and_then(|value| value["schema_version"].as_u64()),
            Some(1)
        );
        assert_eq!(
            versioned_payload("synthetic records", r#"{"schema_version":1}"#).ok(),
            Some(None)
        );
        assert!(matches!(
            versioned_payload("synthetic records", r#"{"schema_version":2}"#),
            Err(LegacyMigrationError::UnsupportedVersion { version: 2, .. })
        ));
        for invalid in ["[]", "not-json", r#"{"schema_version":"one"}"#] {
            assert!(matches!(
                versioned_payload("synthetic records", invalid),
                Err(LegacyMigrationError::MalformedJson { .. })
            ));
        }
        assert!(!MigrationMode::DryRun.applies());
        assert!(MigrationMode::Apply.applies());
    }

    #[test]
    fn task_migration_repairs_future_state_without_executing_missed_runs() {
        let now = Utc
            .with_ymd_and_hms(2026, 9, 3, 12, 0, 0)
            .single()
            .map_or(0, |value| value.timestamp());
        let payload = json!({
            "id":"synthetic-task",
            "chat_id":"synthetic-chat",
            "text":"synthetic reminder",
            "user_name":"synthetic-user",
            "user_id":42,
            "interval_seconds":null,
            "run_date":null,
            "trigger_config":{"type":"cron","hour":20,"minute":30},
            "timezone_offset":-3
        })
        .to_string();
        let change = task_change(payload, now);
        assert!(change.is_ok());
        let Ok(change) = change else { return };
        assert!(
            change
                .document
                .task
                .next_run_at
                .is_some_and(|next| next > now)
        );
        assert_eq!(change.document.task.schedule_anchor_at, Some(now));
        let encoded = serde_json::from_str::<Value>(&change.replacement);
        assert!(encoded.is_ok());
        let Ok(encoded) = encoded else { return };
        assert_eq!(encoded["schema_version"], json!(1));
        assert!(encoded["next_run_at"].is_string());
    }

    #[test]
    fn task_migration_preserves_canonical_next_run() {
        let payload = json!({
            "schema_version":1,
            "id":"synthetic-task",
            "chat_id":"synthetic-chat",
            "text":"synthetic reminder",
            "user_name":"synthetic-user",
            "user_id":42,
            "interval_seconds":600,
            "run_date":null,
            "trigger_config":null,
            "timezone_offset":-3,
            "locale":"es",
            "schedule_anchor_at":"2026-09-03T12:00:00Z",
            "next_run_at":"2026-09-03T12:10:00Z",
            "last_execution_id":null
        })
        .to_string();
        let change = task_change(payload, 1_000);
        assert!(change.is_ok());
        let Ok(change) = change else { return };
        assert_eq!(change.document.task.next_run_at, Some(1_788_437_400));
    }

    #[test]
    fn task_migration_rejects_timestamp_overflow() {
        let payload = r#"{"id":"synthetic-task","chat_id":"synthetic-chat","text":"synthetic","interval_seconds":300}"#.to_owned();
        assert!(matches!(
            task_change(payload, i64::MAX),
            Err(LegacyMigrationError::TaskState(_))
        ));
    }

    #[test]
    fn conditional_writes_reject_concurrent_record_and_task_changes() -> Result<(), Box<dyn Error>>
    {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = integer_server(listener, 0);
        let mut connection = connect(&RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            password: None,
        })?;
        let record_error = apply_changes(
            &mut connection,
            "synthetic records",
            &[Change {
                key: "synthetic:key".to_owned(),
                field: ChangeField::String,
                previous: "old".to_owned(),
                replacement: "new".to_owned(),
            }],
        );
        assert!(matches!(
            record_error,
            Err(LegacyMigrationError::ConcurrentChange { .. })
        ));
        assert!(server.join().is_ok_and(|result| result.is_ok()));

        let task = task_change(
            r#"{"id":"synthetic-task","chat_id":"synthetic-chat","text":"synthetic","interval_seconds":300}"#.to_owned(),
            1_000,
        )?;
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = integer_server(listener, 0);
        let mut connection = connect(&RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            password: None,
        })?;
        assert!(matches!(
            apply_task_change(&mut connection, &task),
            Err(LegacyMigrationError::ConcurrentChange { .. })
        ));
        assert!(server.join().is_ok_and(|result| result.is_ok()));
        Ok(())
    }

    #[test]
    fn redis_apply_versions_records_and_repairs_task_indexes() -> Result<(), Box<dyn Error>> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = migration_server(listener);
        let report = migrate_redis(
            &RedisEndpoint {
                host: "127.0.0.1".to_owned(),
                port,
                password: None,
            },
            MigrationMode::Apply,
            1_788_436_800,
        )?;
        assert_eq!(report.history.migrated, 1);
        assert_eq!(report.chat_members.migrated, 1);
        assert_eq!(report.bot_metadata.migrated, 1);
        assert_eq!(report.tasks_scanned, 1);
        assert_eq!(report.tasks_upgraded, 1);
        assert_eq!(report.task_indexes_rebuilt, 1);
        assert!(server.join().is_ok_and(|result| result.is_ok()));
        Ok(())
    }

    #[test]
    fn redis_dry_run_counts_current_invalid_and_disappearing_records() -> Result<(), Box<dyn Error>>
    {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = inspection_server(listener);
        let report = migrate_redis(
            &RedisEndpoint {
                host: "127.0.0.1".to_owned(),
                port,
                password: None,
            },
            MigrationMode::DryRun,
            1_788_436_800,
        )?;
        for family in [&report.history, &report.chat_members] {
            assert_eq!(family.scanned, 4);
            assert_eq!(family.current, 1);
            assert_eq!(family.candidates, 1);
            assert_eq!(family.migrated, 0);
            assert_eq!(family.malformed, 1);
            assert_eq!(family.unsupported, 1);
        }
        assert_eq!(report.bot_metadata.scanned, 3);
        assert_eq!(report.bot_metadata.current, 1);
        assert_eq!(report.bot_metadata.malformed, 1);
        assert_eq!(report.bot_metadata.unsupported, 1);
        assert_eq!(report.tasks_scanned, 0);
        assert_eq!(report.task_indexes_rebuilt, 0);
        assert!(server.join().is_ok_and(|result| result.is_ok()));
        Ok(())
    }

    #[test]
    fn postgres_migration_is_dry_run_safe_and_idempotent() -> Result<(), Box<dyn Error>> {
        let Some(database_url) = std::env::var("TEST_DATABASE_URL").ok() else {
            return Ok(());
        };
        ChatConfigRepository::new(&database_url).ensure_schema()?;
        let mut client = Client::connect(&database_url, postgres_tls_connector(&database_url)?)?;
        let chat_id = "synthetic-legacy-migration";
        client.execute("DELETE FROM chat_configs WHERE chat_id = $1", &[&chat_id])?;
        client.execute(
            "INSERT INTO chat_configs (chat_id, config) VALUES ($1, $2)",
            &[
                &chat_id,
                &json!({"language":"es","world_cup_goal_alerts":true}),
            ],
        )?;

        let dry_run = migrate_postgres(&database_url, MigrationMode::DryRun)?;
        assert!(dry_run.obsolete_chat_config_fields >= 1);
        assert_eq!(dry_run.removed_chat_config_fields, 0);
        let apply = migrate_postgres(&database_url, MigrationMode::Apply)?;
        assert!(apply.removed_chat_config_fields >= 1);
        let repeated = migrate_postgres(&database_url, MigrationMode::Apply)?;
        assert_eq!(repeated.obsolete_chat_config_fields, 0);
        assert_eq!(repeated.removed_chat_config_fields, 0);
        client.execute("DELETE FROM chat_configs WHERE chat_id = $1", &[&chat_id])?;
        Ok(())
    }
}
