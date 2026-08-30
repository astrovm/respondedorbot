//! Redis string operations used by JSON request caches during migration.

use thiserror::Error;

use crate::redis_connection::{RedisEndpoint, client};

#[derive(Debug, Error)]
pub enum RedisJsonCacheError {
    #[error("Redis JSON-cache operation failed: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("JSON-cache TTL must be non-negative")]
    InvalidTtl,
    #[error("atomic cache locks require a TTL")]
    MissingLockTtl,
}

pub struct RedisJsonCache {
    client: redis::Client,
}

impl RedisJsonCache {
    pub fn new(endpoint: &RedisEndpoint) -> Result<Self, RedisJsonCacheError> {
        Ok(Self {
            client: client(endpoint)?,
        })
    }

    pub fn get(&self, key: &str) -> Result<Option<String>, RedisJsonCacheError> {
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("GET").arg(key).query(&mut connection)?)
    }

    pub fn set(
        &self,
        key: &str,
        value: &str,
        ttl_seconds: Option<i64>,
    ) -> Result<bool, RedisJsonCacheError> {
        let ttl_seconds = ttl_seconds
            .map(|ttl| u64::try_from(ttl).map_err(|_| RedisJsonCacheError::InvalidTtl))
            .transpose()?;
        let mut connection = self.client.get_connection()?;
        if let Some(ttl) = ttl_seconds {
            redis::cmd("SETEX")
                .arg(key)
                .arg(ttl)
                .arg(value)
                .query::<()>(&mut connection)?;
        } else {
            redis::cmd("SET")
                .arg(key)
                .arg(value)
                .query::<()>(&mut connection)?;
        }
        Ok(true)
    }

    pub fn set_if_absent(
        &self,
        key: &str,
        value: &str,
        ttl_seconds: i64,
    ) -> Result<bool, RedisJsonCacheError> {
        let ttl_seconds =
            u64::try_from(ttl_seconds).map_err(|_| RedisJsonCacheError::InvalidTtl)?;
        let mut connection = self.client.get_connection()?;
        let result: Option<String> = redis::cmd("SET")
            .arg(key)
            .arg(value)
            .arg("NX")
            .arg("EX")
            .arg(ttl_seconds)
            .query(&mut connection)?;
        Ok(result.is_some())
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, io::Write, net::TcpListener, thread, time::Duration};

    use super::{RedisJsonCache, RedisJsonCacheError};
    use crate::redis_connection::{RedisEndpoint, test_support::read_command};

    #[test]
    fn speaks_compatible_get_set_and_setex() -> Result<(), Box<dyn Error + Send + Sync>> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<(), Box<dyn Error + Send + Sync>> {
            let exchanges = [
                (
                    vec!["GET", "request_cache:key"],
                    b"$7\r\npayload\r\n".as_slice(),
                ),
                (
                    vec!["SET", "request_cache:key", "persistent"],
                    b"+OK\r\n".as_slice(),
                ),
                (
                    vec!["SETEX", "request_cache:key", "60", "expiring"],
                    b"+OK\r\n".as_slice(),
                ),
            ];
            for (expected, response) in exchanges {
                let (mut stream, _) = listener.accept()?;
                stream.set_read_timeout(Some(Duration::from_secs(2)))?;
                assert_eq!(read_command(&mut stream)?, expected);
                stream.write_all(response)?;
            }
            Ok(())
        });
        let cache = RedisJsonCache::new(&RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            password: None,
        })?;

        assert_eq!(cache.get("request_cache:key")?, Some("payload".to_owned()));
        assert!(cache.set("request_cache:key", "persistent", None)?);
        assert!(cache.set("request_cache:key", "expiring", Some(60))?);
        match server.join() {
            Ok(result) => result?,
            Err(_) => return Err("synthetic Redis server panicked".into()),
        }
        Ok(())
    }

    #[test]
    fn rejects_negative_ttl_before_connecting() -> Result<(), RedisJsonCacheError> {
        let cache = RedisJsonCache::new(&RedisEndpoint {
            host: "invalid.invalid".to_owned(),
            port: 1,
            password: None,
        })?;
        assert!(matches!(
            cache.set("key", "value", Some(-1)),
            Err(RedisJsonCacheError::InvalidTtl)
        ));
        assert!(matches!(
            cache.set_if_absent("key", "value", -1),
            Err(RedisJsonCacheError::InvalidTtl)
        ));
        Ok(())
    }

    #[test]
    fn speaks_compatible_set_nx_ex() -> Result<(), Box<dyn Error + Send + Sync>> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<(), Box<dyn Error + Send + Sync>> {
            let exchanges = [
                (
                    vec!["SET", "refresh:lock", "1", "NX", "EX", "10"],
                    b"+OK\r\n".as_slice(),
                ),
                (
                    vec!["SET", "refresh:lock", "1", "NX", "EX", "10"],
                    b"$-1\r\n".as_slice(),
                ),
            ];
            for (expected, response) in exchanges {
                let (mut stream, _) = listener.accept()?;
                stream.set_read_timeout(Some(Duration::from_secs(2)))?;
                assert_eq!(read_command(&mut stream)?, expected);
                stream.write_all(response)?;
            }
            Ok(())
        });
        let cache = RedisJsonCache::new(&RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            password: None,
        })?;

        assert!(cache.set_if_absent("refresh:lock", "1", 10)?);
        assert!(!cache.set_if_absent("refresh:lock", "1", 10)?);
        match server.join() {
            Ok(result) => result?,
            Err(_) => return Err("synthetic Redis server panicked".into()),
        }
        Ok(())
    }
}
