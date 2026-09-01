//! Redis counter used for per-user group-funded AI message limits.

use thiserror::Error;

use crate::redis_connection::{RedisEndpoint, RedisPool, pool};

pub const CREDITLESS_CAP_TTL_SECONDS: i64 = 3_600;

const REFUND_MARKER_TTL_SECONDS: u64 = 7 * 24 * 60 * 60;
const ADMIT_ONCE_SCRIPT: &str = r#"
local existing = redis.call('GET', KEYS[2])
if existing then
    return tonumber(existing)
end
redis.call('DEL', KEYS[3])
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
local ttl = redis.call('TTL', KEYS[1])
if ttl > 0 then
    redis.call('SET', KEYS[2], count, 'EX', ttl)
end
return count
"#;
const REFUND_ONCE_SCRIPT: &str = r#"
if redis.call('SET', KEYS[3], '1', 'EX', ARGV[1], 'NX') then
    if redis.call('DEL', KEYS[2]) == 1 and redis.call('EXISTS', KEYS[1]) == 1 then
        redis.call('DECR', KEYS[1])
    end
    return 1
end
return 0
"#;

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

    pub fn admit_once(
        &self,
        key: &str,
        operation_id: &str,
        ttl_seconds: i64,
    ) -> Result<i64, RedisCreditlessCapError> {
        let ttl_seconds = u64::try_from(ttl_seconds)
            .ok()
            .filter(|ttl| *ttl > 0)
            .ok_or(RedisCreditlessCapError::InvalidTtl)?;
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("EVAL")
            .arg(ADMIT_ONCE_SCRIPT)
            .arg(3)
            .arg(key)
            .arg(operation_key(operation_id))
            .arg(refund_key(operation_id))
            .arg(ttl_seconds)
            .query(&mut connection)?)
    }

    pub fn refund_once(
        &self,
        key: &str,
        operation_id: &str,
    ) -> Result<bool, RedisCreditlessCapError> {
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("EVAL")
            .arg(REFUND_ONCE_SCRIPT)
            .arg(3)
            .arg(key)
            .arg(operation_key(operation_id))
            .arg(refund_key(operation_id))
            .arg(REFUND_MARKER_TTL_SECONDS)
            .query::<i64>(&mut connection)?
            == 1)
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

fn refund_key(operation_id: &str) -> String {
    format!("creditless_cap_refund:{operation_id}")
}

fn operation_key(operation_id: &str) -> String {
    format!("creditless_cap_operation:{operation_id}")
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
    fn admission_is_idempotent_and_keeps_the_original_expiry()
    -> Result<(), Box<dyn Error + Send + Sync>> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<(), Box<dyn Error + Send + Sync>> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            let first = read_command(&mut stream)?;
            assert_eq!(first.first().map(String::as_str), Some("EVAL"));
            assert_eq!(first.get(2).map(String::as_str), Some("3"));
            assert_eq!(
                first.get(3).map(String::as_str),
                Some("creditless_cap:-42:7")
            );
            assert_eq!(
                first.get(4).map(String::as_str),
                Some("creditless_cap_operation:ai:42:7:first")
            );
            assert_eq!(
                first.get(5).map(String::as_str),
                Some("creditless_cap_refund:ai:42:7:first")
            );
            assert_eq!(first.get(6).map(String::as_str), Some("3600"));
            stream.write_all(b":1\r\n")?;
            let replay = read_command(&mut stream)?;
            assert_eq!(replay.first().map(String::as_str), Some("EVAL"));
            assert_eq!(
                replay.get(4).map(String::as_str),
                Some("creditless_cap_operation:ai:42:7:first")
            );
            stream.write_all(b":1\r\n")?;
            let second = read_command(&mut stream)?;
            assert_eq!(second.first().map(String::as_str), Some("EVAL"));
            assert_eq!(second.get(2).map(String::as_str), Some("3"));
            assert_eq!(
                second.get(4).map(String::as_str),
                Some("creditless_cap_operation:ai:42:7:second")
            );
            stream.write_all(b":2\r\n")?;
            let refund = read_command(&mut stream)?;
            assert_eq!(refund.first().map(String::as_str), Some("EVAL"));
            assert_eq!(refund.get(2).map(String::as_str), Some("3"));
            assert_eq!(
                refund.get(3).map(String::as_str),
                Some("creditless_cap:-42:7")
            );
            assert_eq!(
                refund.get(4).map(String::as_str),
                Some("creditless_cap_operation:ai:42:7:first")
            );
            assert_eq!(
                refund.get(5).map(String::as_str),
                Some("creditless_cap_refund:ai:42:7:first")
            );
            assert_eq!(refund.get(6).map(String::as_str), Some("604800"));
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
        assert_eq!(
            cap.admit_once("creditless_cap:-42:7", "ai:42:7:first", 3_600)?,
            1
        );
        assert_eq!(
            cap.admit_once("creditless_cap:-42:7", "ai:42:7:first", 3_600)?,
            1
        );
        assert_eq!(
            cap.admit_once("creditless_cap:-42:7", "ai:42:7:second", 3_600)?,
            2
        );
        assert!(cap.refund_once("creditless_cap:-42:7", "ai:42:7:first")?);
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
            cap.admit_once("creditless_cap:-42:7", "ai:42:7", 0),
            Err(RedisCreditlessCapError::InvalidTtl)
        ));
        Ok(())
    }
}
