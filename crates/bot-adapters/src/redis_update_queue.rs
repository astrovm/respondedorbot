//! Durable Redis storage for Telegram updates accepted by the polling runtime.

use std::collections::HashMap;

use thiserror::Error;

use crate::redis_connection::{RedisEndpoint, RedisPool, pool};

const UPDATES_KEY: &str = "telegram:updates:pending";
const DEAD_UPDATES_KEY: &str = "telegram:updates:dead";

const QUARANTINE_SCRIPT: &str = r#"
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('HDEL', KEYS[2], ARGV[1])
return 1
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueuedUpdate {
    pub update_id: i64,
    pub payload: String,
}

#[derive(Debug, Error)]
pub enum RedisUpdateQueueError {
    #[error("Redis Telegram update-queue operation failed: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("Redis Telegram update queue contains an invalid update id: {0}")]
    InvalidUpdateId(String),
}

#[derive(Clone)]
pub struct RedisUpdateQueue {
    client: RedisPool,
}

impl RedisUpdateQueue {
    pub fn new(endpoint: &RedisEndpoint) -> Result<Self, RedisUpdateQueueError> {
        Ok(Self {
            client: pool(endpoint)?,
        })
    }

    pub fn insert_update(
        &self,
        update_id: i64,
        payload: &str,
    ) -> Result<bool, RedisUpdateQueueError> {
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("HSETNX")
            .arg(UPDATES_KEY)
            .arg(update_id)
            .arg(payload)
            .query(&mut connection)?)
    }

    pub fn list_updates(&self) -> Result<Vec<QueuedUpdate>, RedisUpdateQueueError> {
        let mut connection = self.client.get_connection()?;
        let updates: HashMap<String, String> = redis::cmd("HGETALL")
            .arg(UPDATES_KEY)
            .query(&mut connection)?;
        let mut updates = updates
            .into_iter()
            .map(|(update_id, payload)| {
                let parsed = update_id
                    .parse::<i64>()
                    .map_err(|_| RedisUpdateQueueError::InvalidUpdateId(update_id))?;
                Ok::<_, RedisUpdateQueueError>(QueuedUpdate {
                    update_id: parsed,
                    payload,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        updates.sort_unstable_by_key(|update| update.update_id);
        Ok(updates)
    }

    pub fn replace_update(
        &self,
        update_id: i64,
        payload: &str,
    ) -> Result<(), RedisUpdateQueueError> {
        let mut connection = self.client.get_connection()?;
        redis::cmd("HSET")
            .arg(UPDATES_KEY)
            .arg(update_id)
            .arg(payload)
            .query::<()>(&mut connection)?;
        Ok(())
    }

    pub fn delete_update(&self, update_id: i64) -> Result<bool, RedisUpdateQueueError> {
        let mut connection = self.client.get_connection()?;
        let deleted: usize = redis::cmd("HDEL")
            .arg(UPDATES_KEY)
            .arg(update_id)
            .query(&mut connection)?;
        Ok(deleted > 0)
    }

    pub fn quarantine_update(
        &self,
        update_id: i64,
        payload: &str,
    ) -> Result<(), RedisUpdateQueueError> {
        let mut connection = self.client.get_connection()?;
        redis::cmd("EVAL")
            .arg(QUARANTINE_SCRIPT)
            .arg(2)
            .arg(DEAD_UPDATES_KEY)
            .arg(UPDATES_KEY)
            .arg(update_id)
            .arg(payload)
            .query::<()>(&mut connection)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, io::Write, net::TcpListener, thread, time::Duration};

    use super::RedisUpdateQueue;
    use crate::redis_connection::{RedisEndpoint, test_support::read_command};

    #[test]
    fn persists_lists_updates_and_moves_failures_to_the_dead_queue()
    -> Result<(), Box<dyn Error + Send + Sync>> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<(), Box<dyn Error + Send + Sync>> {
            let exchanges = [
                ("HSETNX", b":1\r\n".as_slice()),
                (
                    "HGETALL",
                    b"*4\r\n$2\r\n11\r\n$9\r\npayload-b\r\n$2\r\n10\r\n$9\r\npayload-a\r\n"
                        .as_slice(),
                ),
                ("HSET", b":1\r\n".as_slice()),
                ("HDEL", b":1\r\n".as_slice()),
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
        let queue = RedisUpdateQueue::new(&RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            password: None,
        })?;

        assert!(queue.insert_update(10, "payload-a")?);
        assert_eq!(
            queue.list_updates()?,
            [
                super::QueuedUpdate {
                    update_id: 10,
                    payload: "payload-a".to_owned(),
                },
                super::QueuedUpdate {
                    update_id: 11,
                    payload: "payload-b".to_owned(),
                },
            ]
        );
        queue.replace_update(10, "replacement")?;
        assert!(queue.delete_update(10)?);
        queue.quarantine_update(11, "failed")?;

        match server.join() {
            Ok(result) => result?,
            Err(_) => return Err("synthetic Redis server panicked".into()),
        }
        Ok(())
    }
}
