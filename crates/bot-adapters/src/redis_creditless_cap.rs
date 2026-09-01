//! Redis counter used for per-user group-funded AI message limits.

use thiserror::Error;

use crate::redis_connection::{RedisEndpoint, RedisPool, pool};

pub const CREDITLESS_CAP_TTL_SECONDS: i64 = 3_600;

#[derive(Debug, Error)]
pub enum RedisCreditlessCapError {
    #[error("Redis creditless-cap operation failed: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("creditless-cap TTL must be positive")]
    InvalidTtl,
}

pub struct RedisCreditlessCap {
    client: RedisPool,
}

impl RedisCreditlessCap {
    pub fn new(endpoint: &RedisEndpoint) -> Result<Self, RedisCreditlessCapError> {
        Ok(Self {
            client: pool(endpoint)?,
        })
    }

    pub fn increment(&self, key: &str, ttl_seconds: i64) -> Result<i64, RedisCreditlessCapError> {
        let ttl_seconds = u64::try_from(ttl_seconds)
            .ok()
            .filter(|ttl| *ttl > 0)
            .ok_or(RedisCreditlessCapError::InvalidTtl)?;
        let mut connection = self.client.get_connection()?;
        let count: i64 = redis::cmd("INCR").arg(key).query(&mut connection)?;
        if count == 1 {
            redis::cmd("EXPIRE")
                .arg(key)
                .arg(ttl_seconds)
                .query::<()>(&mut connection)?;
        }
        Ok(count)
    }

    pub fn decrement(&self, key: &str) -> Result<i64, RedisCreditlessCapError> {
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("DECR").arg(key).query(&mut connection)?)
    }

    pub fn count(&self, key: &str) -> Result<Option<i64>, RedisCreditlessCapError> {
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("GET").arg(key).query(&mut connection)?)
    }
}

#[must_use]
pub fn creditless_cap_key(origin_chat_id: &str, user_id: i64) -> String {
    format!("creditless_cap:{origin_chat_id}:{user_id}")
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::io::Write;
    use std::net::TcpListener;
    use std::thread;
    use std::time::Duration;

    use super::{
        CREDITLESS_CAP_TTL_SECONDS, RedisCreditlessCap, RedisCreditlessCapError, creditless_cap_key,
    };
    use crate::redis_connection::{RedisEndpoint, test_support::read_command};

    #[test]
    fn key_matches_the_python_hourly_counter_contract() {
        assert_eq!(creditless_cap_key("-10042", 88), "creditless_cap:-10042:88");
        assert_eq!(CREDITLESS_CAP_TTL_SECONDS, 3_600);
    }

    #[test]
    fn first_increment_sets_expiry_and_later_increment_keeps_it()
    -> Result<(), Box<dyn Error + Send + Sync>> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<(), Box<dyn Error + Send + Sync>> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            assert_eq!(read_command(&mut stream)?, ["INCR", "creditless_cap:-42:7"]);
            stream.write_all(b":1\r\n")?;
            assert_eq!(
                read_command(&mut stream)?,
                ["EXPIRE", "creditless_cap:-42:7", "3600"]
            );
            stream.write_all(b":1\r\n")?;
            assert_eq!(read_command(&mut stream)?, ["INCR", "creditless_cap:-42:7"]);
            stream.write_all(b":2\r\n")?;
            assert_eq!(read_command(&mut stream)?, ["DECR", "creditless_cap:-42:7"]);
            stream.write_all(b":1\r\n")?;
            assert_eq!(read_command(&mut stream)?, ["GET", "creditless_cap:-42:7"]);
            stream.write_all(b"$1\r\n1\r\n")?;
            Ok(())
        });
        let cap = RedisCreditlessCap::new(&RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            password: None,
        })?;
        assert_eq!(cap.increment("creditless_cap:-42:7", 3_600)?, 1);
        assert_eq!(cap.increment("creditless_cap:-42:7", 3_600)?, 2);
        assert_eq!(cap.decrement("creditless_cap:-42:7")?, 1);
        assert_eq!(cap.count("creditless_cap:-42:7")?, Some(1));
        match server.join() {
            Ok(result) => result?,
            Err(_) => return Err("synthetic Redis server panicked".into()),
        }
        Ok(())
    }

    #[test]
    fn invalid_ttl_fails_before_connecting() -> Result<(), RedisCreditlessCapError> {
        let cap = RedisCreditlessCap::new(&RedisEndpoint {
            host: "invalid.invalid".to_owned(),
            port: 1,
            password: None,
        })?;
        assert!(matches!(
            cap.increment("creditless_cap:-42:7", 0),
            Err(RedisCreditlessCapError::InvalidTtl)
        ));
        Ok(())
    }
}
