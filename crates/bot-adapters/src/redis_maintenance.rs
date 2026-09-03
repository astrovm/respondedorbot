//! Redis memory policy and TTL repair used by periodic maintenance.

use std::collections::HashMap;

use serde::Serialize;
use thiserror::Error;

use crate::redis_connection::{RedisEndpoint, client};

const CHAT_STATE_TTL: i64 = 30 * 24 * 60 * 60;
const GIPHY_STALE_TTL: i64 = 7 * 24 * 60 * 60;

#[derive(Debug, Error)]
pub enum RedisMaintenanceError {
    #[error("Redis maintenance operation failed: {0}")]
    Redis(#[from] redis::RedisError),
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct RedisMaintenanceResult {
    pub expired_keys: usize,
    pub maxmemory: Option<String>,
    pub maxmemory_policy: Option<String>,
}

pub fn run_redis_maintenance(
    endpoint: &RedisEndpoint,
    maxmemory: &str,
    maxmemory_policy: &str,
) -> Result<RedisMaintenanceResult, RedisMaintenanceError> {
    let redis_client = client(endpoint)?;
    let mut connection = redis_client.get_connection()?;
    redis::cmd("CONFIG")
        .arg("SET")
        .arg("maxmemory")
        .arg(maxmemory)
        .query::<()>(&mut connection)?;
    redis::cmd("CONFIG")
        .arg("SET")
        .arg("maxmemory-policy")
        .arg(maxmemory_policy)
        .query::<()>(&mut connection)?;
    let config: HashMap<String, String> = redis::cmd("CONFIG")
        .arg("GET")
        .arg("maxmemory")
        .arg("maxmemory-policy")
        .query(&mut connection)?;

    let mut expired_keys = 0;
    for (pattern, ttl) in [
        ("giphy_pool_stale:*", GIPHY_STALE_TTL),
        ("chat_history:*", CHAT_STATE_TTL),
    ] {
        for key in scan_keys(&mut connection, Some(pattern))? {
            let current_ttl: i64 = redis::cmd("TTL").arg(&key).query(&mut connection)?;
            if current_ttl == -1 {
                let changed: bool = redis::cmd("EXPIRE")
                    .arg(&key)
                    .arg(ttl)
                    .query(&mut connection)?;
                if changed {
                    expired_keys += 1;
                }
            }
        }
    }

    Ok(RedisMaintenanceResult {
        expired_keys,
        maxmemory: config.get("maxmemory").cloned(),
        maxmemory_policy: config.get("maxmemory-policy").cloned(),
    })
}

fn scan_keys(
    connection: &mut redis::Connection,
    pattern: Option<&str>,
) -> Result<Vec<String>, redis::RedisError> {
    let mut cursor = 0_u64;
    let mut keys = Vec::new();
    loop {
        let mut command = redis::cmd("SCAN");
        command.arg(cursor);
        if let Some(pattern) = pattern {
            command.arg("MATCH").arg(pattern);
        }
        command.arg("COUNT").arg(100);
        let (next_cursor, page): (u64, Vec<String>) = command.query(connection)?;
        keys.extend(page);
        cursor = next_cursor;
        if cursor == 0 {
            return Ok(keys);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, io::Write, net::TcpListener, thread, time::Duration};

    use super::run_redis_maintenance;
    use crate::redis_connection::{RedisEndpoint, test_support::read_command};

    #[test]
    fn configures_policy_and_repairs_ttls() -> Result<(), Box<dyn Error + Send + Sync>> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<(), Box<dyn Error + Send + Sync>> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            let exchanges = vec![
                (vec!["CONFIG", "SET", "maxmemory", "256mb"], "+OK\r\n".to_owned()),
                (
                    vec!["CONFIG", "SET", "maxmemory-policy", "allkeys-lru"],
                    "+OK\r\n".to_owned(),
                ),
                (
                    vec!["CONFIG", "GET", "maxmemory", "maxmemory-policy"],
                    "*4\r\n$9\r\nmaxmemory\r\n$9\r\n268435456\r\n$16\r\nmaxmemory-policy\r\n$11\r\nallkeys-lru\r\n".to_owned(),
                ),
                (
                    vec!["SCAN", "0", "MATCH", "giphy_pool_stale:*", "COUNT", "100"],
                    "*2\r\n$1\r\n0\r\n*1\r\n$19\r\ngiphy_pool_stale:gm\r\n".to_owned(),
                ),
                (vec!["TTL", "giphy_pool_stale:gm"], ":-1\r\n".to_owned()),
                (
                    vec!["EXPIRE", "giphy_pool_stale:gm", "604800"],
                    ":1\r\n".to_owned(),
                ),
                (
                    vec!["SCAN", "0", "MATCH", "chat_history:*", "COUNT", "100"],
                    "*2\r\n$1\r\n0\r\n*1\r\n$16\r\nchat_history:123\r\n".to_owned(),
                ),
                (vec!["TTL", "chat_history:123"], ":100\r\n".to_owned()),
            ];
            for (expected, response) in exchanges {
                assert_eq!(read_command(&mut stream)?, expected);
                stream.write_all(response.as_bytes())?;
            }
            Ok(())
        });

        let result = run_redis_maintenance(
            &RedisEndpoint {
                host: "127.0.0.1".to_owned(),
                port,
                password: None,
            },
            "256mb",
            "allkeys-lru",
        )?;

        assert_eq!(result.expired_keys, 1);
        assert_eq!(result.maxmemory.as_deref(), Some("268435456"));
        assert_eq!(result.maxmemory_policy.as_deref(), Some("allkeys-lru"));
        match server.join() {
            Ok(result) => result?,
            Err(_) => return Err("synthetic Redis server panicked".into()),
        }
        Ok(())
    }
}
