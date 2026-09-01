//! Redis operations for scheduled-task payloads and per-chat indexes.

use thiserror::Error;

use crate::redis_connection::{RedisEndpoint, RedisPool, pool};
use crate::task_record::{
    TaskRecordDocument, TaskRecordError, decode_task_record, encode_task_record,
};

pub const TASK_DUE_INDEX_KEY: &str = "task:due";
pub const TASK_SCHEDULER_OWNER_KEY: &str = "task:scheduler:owner";

const RENEW_LEASE_SCRIPT: &str = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('EXPIRE', KEYS[1], ARGV[2]) else return 0 end";
const RELEASE_LEASE_SCRIPT: &str = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end";
const UPSERT_TASK_SCRIPT: &str = "redis.call('SETEX', KEYS[1], ARGV[1], ARGV[2]); redis.call('ZADD', KEYS[2], ARGV[3], ARGV[4]); redis.call('EXPIRE', KEYS[2], ARGV[1]); redis.call('SETEX', KEYS[3], ARGV[1], '1'); redis.call('ZADD', KEYS[4], ARGV[3], ARGV[4]); return 1";
const COMPLETE_TASK_SCRIPT: &str = "if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end; if redis.call('EXISTS', KEYS[2]) == 0 then redis.call('DEL', KEYS[1]); return 0 end; if ARGV[2] == '' then redis.call('DEL', KEYS[2]); redis.call('ZREM', KEYS[3], ARGV[3]); redis.call('ZREM', KEYS[4], ARGV[3]); else redis.call('SETEX', KEYS[2], ARGV[4], ARGV[2]); redis.call('ZADD', KEYS[3], ARGV[5], ARGV[3]); redis.call('EXPIRE', KEYS[3], ARGV[4]); redis.call('ZADD', KEYS[4], ARGV[5], ARGV[3]); end; redis.call('DEL', KEYS[1]); return 1";
const CANCEL_TASK_SCRIPT: &str = "local removed = redis.call('DEL', KEYS[1]); redis.call('ZREM', KEYS[2], ARGV[1]); redis.call('ZREM', KEYS[3], ARGV[1]); return removed";

#[derive(Debug, Error)]
pub enum RedisTaskStoreError {
    #[error("Redis task-store operation failed: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("task-store TTL must be positive")]
    InvalidTtl,
    #[error("task-store limit must be positive")]
    InvalidLimit,
    #[error("task-store score must be finite")]
    InvalidScore,
    #[error("canonical recurring task has no next-run state")]
    MissingNextRun,
    #[error(transparent)]
    Record(#[from] TaskRecordError),
}

#[derive(Clone)]
pub struct RedisTaskStore {
    client: RedisPool,
}

impl RedisTaskStore {
    pub fn new(endpoint: &RedisEndpoint) -> Result<Self, RedisTaskStoreError> {
        Ok(Self {
            client: pool(endpoint)?,
        })
    }

    pub fn get(&self, key: &str) -> Result<Option<String>, RedisTaskStoreError> {
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("GET").arg(key).query(&mut connection)?)
    }

    pub fn setex(
        &self,
        key: &str,
        ttl_seconds: i64,
        value: &str,
    ) -> Result<bool, RedisTaskStoreError> {
        let ttl_seconds = positive_ttl(ttl_seconds)?;
        let mut connection = self.client.get_connection()?;
        redis::cmd("SETEX")
            .arg(key)
            .arg(ttl_seconds)
            .arg(value)
            .query::<()>(&mut connection)?;
        Ok(true)
    }

    pub fn delete(&self, key: &str) -> Result<usize, RedisTaskStoreError> {
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("DEL").arg(key).query(&mut connection)?)
    }

    pub fn zadd(&self, key: &str, member: &str, score: f64) -> Result<usize, RedisTaskStoreError> {
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("ZADD")
            .arg(key)
            .arg(score)
            .arg(member)
            .query(&mut connection)?)
    }

    pub fn expire(&self, key: &str, ttl_seconds: i64) -> Result<bool, RedisTaskStoreError> {
        let ttl_seconds = positive_ttl(ttl_seconds)?;
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("EXPIRE")
            .arg(key)
            .arg(ttl_seconds)
            .query(&mut connection)?)
    }

    pub fn zrem(&self, key: &str, members: &[String]) -> Result<usize, RedisTaskStoreError> {
        if members.is_empty() {
            return Ok(0);
        }
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("ZREM")
            .arg(key)
            .arg(members)
            .query(&mut connection)?)
    }

    pub fn scan(&self, pattern: &str) -> Result<Vec<String>, RedisTaskStoreError> {
        let mut connection = self.client.get_connection()?;
        let mut cursor = 0_u64;
        let mut keys = Vec::new();
        loop {
            let (next_cursor, page): (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(pattern)
                .arg("COUNT")
                .arg(100)
                .query(&mut connection)?;
            keys.extend(page);
            cursor = next_cursor;
            if cursor == 0 {
                return Ok(keys);
            }
        }
    }

    pub fn zrange(&self, key: &str) -> Result<Vec<String>, RedisTaskStoreError> {
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("ZRANGE")
            .arg(key)
            .arg(0)
            .arg(-1)
            .query(&mut connection)?)
    }

    pub fn mget(&self, keys: &[String]) -> Result<Vec<Option<String>>, RedisTaskStoreError> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("MGET").arg(keys).query(&mut connection)?)
    }

    pub fn due_task_ids(&self, now: f64, limit: usize) -> Result<Vec<String>, RedisTaskStoreError> {
        if !now.is_finite() {
            return Err(RedisTaskStoreError::InvalidScore);
        }
        if limit == 0 {
            return Err(RedisTaskStoreError::InvalidLimit);
        }
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("ZRANGEBYSCORE")
            .arg(TASK_DUE_INDEX_KEY)
            .arg("-inf")
            .arg(now)
            .arg("LIMIT")
            .arg(0)
            .arg(limit)
            .query(&mut connection)?)
    }

    pub fn acquire_lease(
        &self,
        key: &str,
        token: &str,
        ttl_seconds: i64,
    ) -> Result<bool, RedisTaskStoreError> {
        let ttl_seconds = positive_ttl(ttl_seconds)?;
        let mut connection = self.client.get_connection()?;
        let result: Option<String> = redis::cmd("SET")
            .arg(key)
            .arg(token)
            .arg("NX")
            .arg("EX")
            .arg(ttl_seconds)
            .query(&mut connection)?;
        Ok(result.is_some())
    }

    pub fn renew_lease(
        &self,
        key: &str,
        token: &str,
        ttl_seconds: i64,
    ) -> Result<bool, RedisTaskStoreError> {
        let ttl_seconds = positive_ttl(ttl_seconds)?;
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("EVAL")
            .arg(RENEW_LEASE_SCRIPT)
            .arg(1)
            .arg(key)
            .arg(token)
            .arg(ttl_seconds)
            .query::<i64>(&mut connection)?
            == 1)
    }

    pub fn release_lease(&self, key: &str, token: &str) -> Result<bool, RedisTaskStoreError> {
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("EVAL")
            .arg(RELEASE_LEASE_SCRIPT)
            .arg(1)
            .arg(key)
            .arg(token)
            .query::<i64>(&mut connection)?
            == 1)
    }

    pub fn upsert_task(
        &self,
        task_id: &str,
        chat_id: &str,
        payload: &str,
        next_run_score: f64,
        ttl_seconds: i64,
    ) -> Result<bool, RedisTaskStoreError> {
        if !next_run_score.is_finite() {
            return Err(RedisTaskStoreError::InvalidScore);
        }
        let ttl_seconds = positive_ttl(ttl_seconds)?;
        let data_key = format!("task:data:{task_id}");
        let chat_key = format!("task:chat:{chat_id}");
        let marker_key = format!("{chat_key}:indexed");
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("EVAL")
            .arg(UPSERT_TASK_SCRIPT)
            .arg(4)
            .arg(data_key)
            .arg(chat_key)
            .arg(marker_key)
            .arg(TASK_DUE_INDEX_KEY)
            .arg(ttl_seconds)
            .arg(payload)
            .arg(next_run_score)
            .arg(task_id)
            .query::<i64>(&mut connection)?
            == 1)
    }

    pub fn complete_occurrence(
        &self,
        occurrence: &TaskOccurrenceCompletion<'_>,
    ) -> Result<bool, RedisTaskStoreError> {
        if !occurrence.next_run_score.is_finite() {
            return Err(RedisTaskStoreError::InvalidScore);
        }
        let ttl_seconds = positive_ttl(occurrence.ttl_seconds)?;
        let claim_key = task_claim_key(occurrence.task_id, occurrence.execution_id);
        let data_key = format!("task:data:{}", occurrence.task_id);
        let chat_key = format!("task:chat:{}", occurrence.chat_id);
        let payload = occurrence.next_payload.unwrap_or_default();
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("EVAL")
            .arg(COMPLETE_TASK_SCRIPT)
            .arg(4)
            .arg(claim_key)
            .arg(data_key)
            .arg(chat_key)
            .arg(TASK_DUE_INDEX_KEY)
            .arg(occurrence.claim_token)
            .arg(payload)
            .arg(occurrence.task_id)
            .arg(ttl_seconds)
            .arg(occurrence.next_run_score)
            .query::<i64>(&mut connection)?
            == 1)
    }

    pub fn cancel_task(&self, task_id: &str, chat_id: &str) -> Result<bool, RedisTaskStoreError> {
        let data_key = format!("task:data:{task_id}");
        let chat_key = format!("task:chat:{chat_id}");
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("EVAL")
            .arg(CANCEL_TASK_SCRIPT)
            .arg(3)
            .arg(data_key)
            .arg(chat_key)
            .arg(TASK_DUE_INDEX_KEY)
            .arg(task_id)
            .query::<i64>(&mut connection)?
            > 0)
    }

    pub fn load_task(
        &self,
        task_id: &str,
    ) -> Result<Option<TaskRecordDocument>, RedisTaskStoreError> {
        self.get(&format!("task:data:{task_id}"))?
            .map(|payload| decode_task_record(&payload).map_err(Into::into))
            .transpose()
    }

    pub fn list_chat_tasks(
        &self,
        chat_id: &str,
    ) -> Result<Vec<TaskRecordDocument>, RedisTaskStoreError> {
        let index_key = format!("task:chat:{chat_id}");
        let task_ids = self.zrange(&index_key)?;
        let payloads = self.mget(
            &task_ids
                .iter()
                .map(|task_id| format!("task:data:{task_id}"))
                .collect::<Vec<_>>(),
        )?;
        let mut tasks = Vec::new();
        let mut missing_ids = Vec::new();
        for (task_id, payload) in task_ids.iter().zip(payloads) {
            let Some(payload) = payload else {
                missing_ids.push(task_id.clone());
                continue;
            };
            let document = decode_task_record(&payload)?;
            if document.task.chat_id == chat_id {
                tasks.push(document);
            }
        }
        let _removed = self.zrem(&index_key, &missing_ids)?;
        Ok(tasks)
    }

    pub fn save_task(
        &self,
        document: &TaskRecordDocument,
        ttl_seconds: i64,
    ) -> Result<bool, RedisTaskStoreError> {
        let next_run_at = document
            .task
            .next_run_at
            .ok_or(RedisTaskStoreError::MissingNextRun)?;
        self.upsert_task(
            document.task.id.as_str(),
            &document.task.chat_id,
            &encode_task_record(document)?,
            next_run_at as f64,
            ttl_seconds,
        )
    }

    pub fn claim_occurrence(
        &self,
        task_id: &str,
        execution_id: &str,
        claim_token: &str,
        ttl_seconds: i64,
    ) -> Result<bool, RedisTaskStoreError> {
        self.acquire_lease(
            &task_claim_key(task_id, execution_id),
            claim_token,
            ttl_seconds,
        )
    }

    pub fn release_occurrence(
        &self,
        task_id: &str,
        execution_id: &str,
        claim_token: &str,
    ) -> Result<bool, RedisTaskStoreError> {
        self.release_lease(&task_claim_key(task_id, execution_id), claim_token)
    }

    pub fn remove_due_task_id(&self, task_id: &str) -> Result<bool, RedisTaskStoreError> {
        Ok(self.zrem(TASK_DUE_INDEX_KEY, &[task_id.to_owned()])? > 0)
    }
}

fn positive_ttl(ttl_seconds: i64) -> Result<u64, RedisTaskStoreError> {
    u64::try_from(ttl_seconds)
        .ok()
        .filter(|ttl| *ttl > 0)
        .ok_or(RedisTaskStoreError::InvalidTtl)
}

pub struct TaskOccurrenceCompletion<'a> {
    pub task_id: &'a str,
    pub chat_id: &'a str,
    pub execution_id: &'a str,
    pub claim_token: &'a str,
    pub next_payload: Option<&'a str>,
    pub next_run_score: f64,
    pub ttl_seconds: i64,
}

#[must_use]
pub fn task_claim_key(task_id: &str, execution_id: &str) -> String {
    format!("task:claim:{task_id}:{execution_id}")
}

#[cfg(test)]
mod tests {
    use std::{error::Error, io::Write, net::TcpListener, thread, time::Duration};

    use super::{
        CANCEL_TASK_SCRIPT, COMPLETE_TASK_SCRIPT, RELEASE_LEASE_SCRIPT, RENEW_LEASE_SCRIPT,
        RedisTaskStore, RedisTaskStoreError, TASK_DUE_INDEX_KEY, TaskOccurrenceCompletion,
        UPSERT_TASK_SCRIPT, task_claim_key,
    };
    use crate::redis_connection::{RedisEndpoint, test_support::read_command};

    #[test]
    fn rejects_nonpositive_ttls_before_connecting() -> Result<(), RedisTaskStoreError> {
        let store = RedisTaskStore::new(&RedisEndpoint {
            host: "invalid.invalid".to_owned(),
            port: 1,
            password: None,
        })?;
        assert!(matches!(
            store.setex("task:data:1", -1, "{}"),
            Err(RedisTaskStoreError::InvalidTtl)
        ));
        assert!(matches!(
            store.setex("task:data:1", 0, "{}"),
            Err(RedisTaskStoreError::InvalidTtl)
        ));
        assert!(matches!(
            store.expire("task:chat:1", -1),
            Err(RedisTaskStoreError::InvalidTtl)
        ));
        assert!(matches!(
            store.acquire_lease("lease", "token", -1),
            Err(RedisTaskStoreError::InvalidTtl)
        ));
        assert!(matches!(
            store.acquire_lease("lease", "token", 0),
            Err(RedisTaskStoreError::InvalidTtl)
        ));
        assert!(matches!(
            store.renew_lease("lease", "token", -1),
            Err(RedisTaskStoreError::InvalidTtl)
        ));
        assert!(matches!(
            store.due_task_ids(1.0, 0),
            Err(RedisTaskStoreError::InvalidLimit)
        ));
        assert!(matches!(
            store.due_task_ids(f64::NAN, 1),
            Err(RedisTaskStoreError::InvalidScore)
        ));
        assert!(matches!(
            store.upsert_task("t1", "c1", "{}", f64::INFINITY, 60),
            Err(RedisTaskStoreError::InvalidScore)
        ));
        Ok(())
    }

    #[test]
    fn preserves_task_payload_and_index_commands() -> Result<(), Box<dyn Error + Send + Sync>> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<(), Box<dyn Error + Send + Sync>> {
            let exchanges = [
                (vec!["GET", "task:data:t1"], "$2\r\n{}\r\n"),
                (vec!["SETEX", "task:data:t1", "60", "{}"], "+OK\r\n"),
                (vec!["DEL", "task:data:t1"], ":1\r\n"),
                (vec!["ZADD", "task:chat:c1", "42.5", "t1"], ":1\r\n"),
                (vec!["EXPIRE", "task:chat:c1", "60"], ":1\r\n"),
                (vec!["ZREM", "task:chat:c1", "t1", "t2"], ":2\r\n"),
                (
                    vec!["SCAN", "0", "MATCH", "task:data:*", "COUNT", "100"],
                    "*2\r\n$1\r\n0\r\n*1\r\n$12\r\ntask:data:t1\r\n",
                ),
                (
                    vec!["ZRANGE", "task:chat:c1", "0", "-1"],
                    "*1\r\n$2\r\nt1\r\n",
                ),
                (
                    vec!["MGET", "task:data:t1", "task:data:t2"],
                    "*2\r\n$2\r\n{}\r\n$-1\r\n",
                ),
            ];
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            for (expected, response) in exchanges {
                assert_eq!(read_command(&mut stream)?, expected);
                stream.write_all(response.as_bytes())?;
            }
            Ok(())
        });
        let store = RedisTaskStore::new(&RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            password: None,
        })?;

        assert_eq!(store.get("task:data:t1")?.as_deref(), Some("{}"));
        assert!(store.setex("task:data:t1", 60, "{}")?);
        assert_eq!(store.delete("task:data:t1")?, 1);
        assert_eq!(store.zadd("task:chat:c1", "t1", 42.5)?, 1);
        assert!(store.expire("task:chat:c1", 60)?);
        assert_eq!(store.zrem("task:chat:c1", &["t1".into(), "t2".into()])?, 2);
        assert_eq!(store.scan("task:data:*")?, ["task:data:t1"]);
        assert_eq!(store.zrange("task:chat:c1")?, ["t1"]);
        assert_eq!(
            store.mget(&["task:data:t1".into(), "task:data:t2".into()])?,
            [Some("{}".into()), None]
        );
        match server.join() {
            Ok(result) => result?,
            Err(_) => return Err("synthetic Redis server panicked".into()),
        }
        Ok(())
    }

    #[test]
    fn claim_keys_are_stable() {
        assert_eq!(
            task_claim_key("t1", "t1:1700000000"),
            "task:claim:t1:t1:1700000000"
        );
    }

    #[test]
    fn executes_due_lease_and_atomic_state_commands() -> Result<(), Box<dyn Error + Send + Sync>> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<(), Box<dyn Error + Send + Sync>> {
            let exchanges = [
                (
                    vec![
                        "ZRANGEBYSCORE",
                        TASK_DUE_INDEX_KEY,
                        "-inf",
                        "42.5",
                        "LIMIT",
                        "0",
                        "10",
                    ],
                    "*1\r\n$2\r\nt1\r\n",
                ),
                (
                    vec!["SET", "task:scheduler:owner", "owner", "NX", "EX", "30"],
                    "+OK\r\n",
                ),
                (
                    vec![
                        "EVAL",
                        RENEW_LEASE_SCRIPT,
                        "1",
                        "task:scheduler:owner",
                        "owner",
                        "30",
                    ],
                    ":1\r\n",
                ),
                (
                    vec![
                        "EVAL",
                        RELEASE_LEASE_SCRIPT,
                        "1",
                        "task:scheduler:owner",
                        "owner",
                    ],
                    ":1\r\n",
                ),
                (
                    vec![
                        "EVAL",
                        UPSERT_TASK_SCRIPT,
                        "4",
                        "task:data:t1",
                        "task:chat:c1",
                        "task:chat:c1:indexed",
                        TASK_DUE_INDEX_KEY,
                        "60",
                        "{}",
                        "42.5",
                        "t1",
                    ],
                    ":1\r\n",
                ),
                (
                    vec![
                        "EVAL",
                        COMPLETE_TASK_SCRIPT,
                        "4",
                        "task:claim:t1:t1:42",
                        "task:data:t1",
                        "task:chat:c1",
                        TASK_DUE_INDEX_KEY,
                        "claim",
                        "{\"next\":true}",
                        "t1",
                        "60",
                        "100.0",
                    ],
                    ":1\r\n",
                ),
                (
                    vec![
                        "EVAL",
                        RELEASE_LEASE_SCRIPT,
                        "1",
                        "task:claim:t1:t1:42",
                        "claim",
                    ],
                    ":1\r\n",
                ),
                (vec!["ZREM", TASK_DUE_INDEX_KEY, "stale"], ":1\r\n"),
                (
                    vec![
                        "EVAL",
                        CANCEL_TASK_SCRIPT,
                        "3",
                        "task:data:t1",
                        "task:chat:c1",
                        TASK_DUE_INDEX_KEY,
                        "t1",
                    ],
                    ":1\r\n",
                ),
            ];
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            for (expected, response) in exchanges {
                assert_eq!(read_command(&mut stream)?, expected);
                stream.write_all(response.as_bytes())?;
            }
            Ok(())
        });
        let store = RedisTaskStore::new(&RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            password: None,
        })?;

        assert_eq!(store.due_task_ids(42.5, 10)?, ["t1"]);
        assert!(store.acquire_lease("task:scheduler:owner", "owner", 30)?);
        assert!(store.renew_lease("task:scheduler:owner", "owner", 30)?);
        assert!(store.release_lease("task:scheduler:owner", "owner")?);
        assert!(store.upsert_task("t1", "c1", "{}", 42.5, 60)?);
        assert!(store.complete_occurrence(&TaskOccurrenceCompletion {
            task_id: "t1",
            chat_id: "c1",
            execution_id: "t1:42",
            claim_token: "claim",
            next_payload: Some("{\"next\":true}"),
            next_run_score: 100.0,
            ttl_seconds: 60,
        })?);
        assert!(store.release_occurrence("t1", "t1:42", "claim")?);
        assert!(store.remove_due_task_id("stale")?);
        assert!(store.cancel_task("t1", "c1")?);
        match server.join() {
            Ok(result) => result?,
            Err(_) => return Err("synthetic Redis server panicked".into()),
        }
        Ok(())
    }
}
