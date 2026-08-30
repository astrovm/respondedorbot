//! Redis operations for scheduled-task payloads and per-chat indexes.

use thiserror::Error;

use crate::redis_connection::{RedisEndpoint, client};

#[derive(Debug, Error)]
pub enum RedisTaskStoreError {
    #[error("Redis task-store operation failed: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("task-store TTL must be non-negative")]
    InvalidTtl,
}

pub struct RedisTaskStore {
    client: redis::Client,
}

impl RedisTaskStore {
    pub fn new(endpoint: &RedisEndpoint) -> Result<Self, RedisTaskStoreError> {
        Ok(Self {
            client: client(endpoint)?,
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
        let ttl_seconds =
            u64::try_from(ttl_seconds).map_err(|_| RedisTaskStoreError::InvalidTtl)?;
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
        let ttl_seconds =
            u64::try_from(ttl_seconds).map_err(|_| RedisTaskStoreError::InvalidTtl)?;
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
}

#[cfg(test)]
mod tests {
    use std::{error::Error, io::Write, net::TcpListener, thread, time::Duration};

    use super::{RedisTaskStore, RedisTaskStoreError};
    use crate::redis_connection::{RedisEndpoint, test_support::read_command};

    #[test]
    fn rejects_negative_ttls_before_connecting() -> Result<(), RedisTaskStoreError> {
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
            store.expire("task:chat:1", -1),
            Err(RedisTaskStoreError::InvalidTtl)
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
            for (expected, response) in exchanges {
                let (mut stream, _) = listener.accept()?;
                stream.set_read_timeout(Some(Duration::from_secs(2)))?;
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
}
