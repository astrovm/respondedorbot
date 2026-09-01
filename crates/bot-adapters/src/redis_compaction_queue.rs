//! Durable Redis storage and lease operations for memory-compaction jobs.

use std::collections::HashMap;

use serde::Serialize;
use thiserror::Error;

use crate::redis_connection::{RedisEndpoint, RedisPool, pool};

const JOBS_KEY: &str = "memory:compaction:jobs";
const DEAD_JOBS_KEY: &str = "memory:compaction:dead_jobs";
const LOCK_PREFIX: &str = "memory:compaction:lock:";

const RELEASE_LOCK_SCRIPT: &str = r#"
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"#;

const QUARANTINE_SCRIPT: &str = r#"
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('HDEL', KEYS[2], ARGV[3])
return 1
"#;

#[derive(Debug, Error)]
pub enum RedisCompactionQueueError {
    #[error("Redis compaction-queue operation failed: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("compaction lock TTL must be positive")]
    InvalidLockTtl,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct QueueJob {
    pub chat_id: String,
    pub payload: String,
}

pub struct RedisCompactionQueue {
    client: RedisPool,
}

impl RedisCompactionQueue {
    pub fn new(endpoint: &RedisEndpoint) -> Result<Self, RedisCompactionQueueError> {
        Ok(Self {
            client: pool(endpoint)?,
        })
    }

    pub fn job_exists(&self, chat_id: &str) -> Result<bool, RedisCompactionQueueError> {
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("HEXISTS")
            .arg(JOBS_KEY)
            .arg(chat_id)
            .query(&mut connection)?)
    }

    pub fn insert_job(
        &self,
        chat_id: &str,
        payload: &str,
    ) -> Result<bool, RedisCompactionQueueError> {
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("HSETNX")
            .arg(JOBS_KEY)
            .arg(chat_id)
            .arg(payload)
            .query(&mut connection)?)
    }

    pub fn list_jobs(&self) -> Result<Vec<QueueJob>, RedisCompactionQueueError> {
        let mut connection = self.client.get_connection()?;
        let jobs: HashMap<String, String> =
            redis::cmd("HGETALL").arg(JOBS_KEY).query(&mut connection)?;
        let mut jobs = jobs
            .into_iter()
            .map(|(chat_id, payload)| QueueJob { chat_id, payload })
            .collect::<Vec<_>>();
        jobs.sort_unstable_by(|left, right| left.chat_id.cmp(&right.chat_id));
        Ok(jobs)
    }

    pub fn replace_job(
        &self,
        chat_id: &str,
        payload: &str,
    ) -> Result<(), RedisCompactionQueueError> {
        let mut connection = self.client.get_connection()?;
        redis::cmd("HSET")
            .arg(JOBS_KEY)
            .arg(chat_id)
            .arg(payload)
            .query::<()>(&mut connection)?;
        Ok(())
    }

    pub fn delete_job(&self, chat_id: &str) -> Result<bool, RedisCompactionQueueError> {
        let mut connection = self.client.get_connection()?;
        let deleted: usize = redis::cmd("HDEL")
            .arg(JOBS_KEY)
            .arg(chat_id)
            .query(&mut connection)?;
        Ok(deleted > 0)
    }

    pub fn acquire_lock(
        &self,
        chat_id: &str,
        token: &str,
        ttl_seconds: i64,
    ) -> Result<bool, RedisCompactionQueueError> {
        let ttl_seconds = u64::try_from(ttl_seconds)
            .ok()
            .filter(|ttl| *ttl > 0)
            .ok_or(RedisCompactionQueueError::InvalidLockTtl)?;
        let mut connection = self.client.get_connection()?;
        let result: Option<String> = redis::cmd("SET")
            .arg(format!("{LOCK_PREFIX}{chat_id}"))
            .arg(token)
            .arg("NX")
            .arg("EX")
            .arg(ttl_seconds)
            .query(&mut connection)?;
        Ok(result.is_some())
    }

    pub fn release_lock(
        &self,
        chat_id: &str,
        token: &str,
    ) -> Result<bool, RedisCompactionQueueError> {
        let mut connection = self.client.get_connection()?;
        let deleted: usize = redis::cmd("EVAL")
            .arg(RELEASE_LOCK_SCRIPT)
            .arg(1)
            .arg(format!("{LOCK_PREFIX}{chat_id}"))
            .arg(token)
            .query(&mut connection)?;
        Ok(deleted > 0)
    }

    pub fn quarantine_job(
        &self,
        chat_id: &str,
        dead_job_id: &str,
        dead_payload: &str,
    ) -> Result<bool, RedisCompactionQueueError> {
        let mut connection = self.client.get_connection()?;
        let deleted: usize = redis::cmd("EVAL")
            .arg(QUARANTINE_SCRIPT)
            .arg(2)
            .arg(DEAD_JOBS_KEY)
            .arg(JOBS_KEY)
            .arg(dead_job_id)
            .arg(dead_payload)
            .arg(chat_id)
            .query(&mut connection)?;
        Ok(deleted > 0)
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, io::Write, net::TcpListener, thread, time::Duration};

    use super::{RedisCompactionQueue, RedisCompactionQueueError};
    use crate::redis_connection::{RedisEndpoint, test_support::read_command};

    #[test]
    fn operates_on_existing_queue_keys_and_atomic_leases()
    -> Result<(), Box<dyn Error + Send + Sync>> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<(), Box<dyn Error + Send + Sync>> {
            let exchanges = [
                ("HEXISTS", b":1\r\n".as_slice()),
                ("HSETNX", b":1\r\n".as_slice()),
                (
                    "HGETALL",
                    b"*4\r\n$6\r\nchat-b\r\n$9\r\npayload-b\r\n$6\r\nchat-a\r\n$9\r\npayload-a\r\n"
                        .as_slice(),
                ),
                ("HSET", b":1\r\n".as_slice()),
                ("HDEL", b":1\r\n".as_slice()),
                ("SET", b"+OK\r\n".as_slice()),
                ("EVAL", b":1\r\n".as_slice()),
                ("EVAL", b":1\r\n".as_slice()),
            ];
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            for (expected, response) in exchanges {
                let command = read_command(&mut stream)?;
                assert_eq!(command.first().map(String::as_str), Some(expected));
                stream.write_all(response)?;
            }
            Ok(())
        });
        let queue = RedisCompactionQueue::new(&RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            password: None,
        })?;

        assert!(queue.job_exists("chat-a")?);
        assert!(queue.insert_job("chat-a", "payload-a")?);
        assert_eq!(
            queue.list_jobs()?,
            [
                super::QueueJob {
                    chat_id: "chat-a".to_owned(),
                    payload: "payload-a".to_owned(),
                },
                super::QueueJob {
                    chat_id: "chat-b".to_owned(),
                    payload: "payload-b".to_owned(),
                },
            ]
        );
        queue.replace_job("chat-a", "replacement")?;
        assert!(queue.delete_job("chat-a")?);
        assert!(queue.acquire_lock("chat-a", "token", 3_600)?);
        assert!(queue.release_lock("chat-a", "token")?);
        assert!(queue.quarantine_job("chat-a", "dead-id", "dead-payload")?);

        match server.join() {
            Ok(result) => result?,
            Err(_) => return Err("synthetic Redis server panicked".into()),
        }
        Ok(())
    }

    #[test]
    fn rejects_nonpositive_lock_ttl_before_connecting() -> Result<(), RedisCompactionQueueError> {
        let queue = RedisCompactionQueue::new(&RedisEndpoint {
            host: "invalid.invalid".to_owned(),
            port: 1,
            password: None,
        })?;
        assert!(matches!(
            queue.acquire_lock("chat", "token", 0),
            Err(RedisCompactionQueueError::InvalidLockTtl)
        ));
        Ok(())
    }
}
