//! Shared synchronous Redis connection boundary.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};
use std::time::{Duration, Instant};

use redis::{Commands, ConnectionLike, IntoConnectionInfo, RedisConnectionInfo};

use crate::idle_pool::{IdleList, MAX_IDLE_AGE};

const MAX_IDLE_CONNECTIONS: usize = 16;
/// Bound how long a worker can hang on an unreachable or stalled Redis
/// instead of waiting for the operating system's TCP timeout.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
const IO_TIMEOUT: Duration = Duration::from_secs(10);

struct RedisPoolInner {
    client: redis::Client,
    idle: Mutex<IdleList<redis::Connection>>,
    max_idle_connections: usize,
}

#[derive(Clone)]
pub(crate) struct RedisPool {
    inner: Arc<RedisPoolInner>,
}

pub(crate) struct RedisPooledConnection {
    connection: Option<redis::Connection>,
    pool: Arc<RedisPoolInner>,
}

impl RedisPool {
    pub(crate) fn get_connection(&self) -> redis::RedisResult<RedisPooledConnection> {
        let connection = self
            .inner
            .idle
            .lock()
            .ok()
            .and_then(|mut idle| idle.pop_fresh(Instant::now(), MAX_IDLE_AGE))
            .map_or_else(|| open_connection(&self.inner.client), Ok)?;
        Ok(RedisPooledConnection {
            connection: Some(connection),
            pool: self.inner.clone(),
        })
    }
}

impl Deref for RedisPooledConnection {
    type Target = redis::Connection;

    #[allow(
        clippy::expect_used,
        reason = "the connection is only taken by drop, after which it cannot be dereferenced"
    )]
    fn deref(&self) -> &Self::Target {
        self.connection
            .as_ref()
            .expect("pooled Redis connection is present until drop")
    }
}

impl DerefMut for RedisPooledConnection {
    #[allow(
        clippy::expect_used,
        reason = "the connection is only taken by drop, after which it cannot be dereferenced"
    )]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.connection
            .as_mut()
            .expect("pooled Redis connection is present until drop")
    }
}

impl Drop for RedisPooledConnection {
    fn drop(&mut self) {
        if let Some(connection) = self.connection.take()
            && connection.is_open()
            && let Ok(mut idle) = self.pool.idle.lock()
            && idle.len() < self.pool.max_idle_connections
        {
            idle.push(Instant::now(), connection);
        }
    }
}

#[derive(Clone)]
pub struct RedisEndpoint {
    pub host: String,
    pub port: u16,
    pub password: Option<String>,
}

pub trait RedisStringCommands {
    fn get_text(&mut self, key: &str) -> redis::RedisResult<Option<String>>;
    fn set_text(&mut self, key: &str, value: &str, ttl_seconds: u64) -> redis::RedisResult<()>;
}

impl RedisStringCommands for redis::Connection {
    fn get_text(&mut self, key: &str) -> redis::RedisResult<Option<String>> {
        self.get(key)
    }

    fn set_text(&mut self, key: &str, value: &str, ttl_seconds: u64) -> redis::RedisResult<()> {
        self.set_ex(key, value, ttl_seconds)
    }
}

impl RedisStringCommands for RedisPooledConnection {
    fn get_text(&mut self, key: &str) -> redis::RedisResult<Option<String>> {
        self.get(key)
    }

    fn set_text(&mut self, key: &str, value: &str, ttl_seconds: u64) -> redis::RedisResult<()> {
        self.set_ex(key, value, ttl_seconds)
    }
}

pub(crate) fn connect(endpoint: &RedisEndpoint) -> redis::RedisResult<RedisPooledConnection> {
    pool(endpoint)?.get_connection()
}

pub(crate) fn pool(endpoint: &RedisEndpoint) -> redis::RedisResult<RedisPool> {
    static POOLS: OnceLock<Mutex<HashMap<String, Weak<RedisPoolInner>>>> = OnceLock::new();
    let key = format!(
        "{}:{}:{}",
        endpoint.host,
        endpoint.port,
        endpoint.password.as_deref().unwrap_or_default()
    );
    let pools = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    // The map is only read and extended while locked, so a poisoned lock
    // still guards a consistent map.
    let mut pools = pools.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(inner) = pools.get(&key).and_then(Weak::upgrade) {
        return Ok(RedisPool { inner });
    }
    let inner = Arc::new(RedisPoolInner {
        client: client(endpoint)?,
        idle: Mutex::new(IdleList::new()),
        max_idle_connections: MAX_IDLE_CONNECTIONS,
    });
    pools.insert(key, Arc::downgrade(&inner));
    Ok(RedisPool { inner })
}

fn open_connection(client: &redis::Client) -> redis::RedisResult<redis::Connection> {
    let connection = client.get_connection_with_timeout(CONNECT_TIMEOUT)?;
    connection.set_read_timeout(Some(IO_TIMEOUT))?;
    connection.set_write_timeout(Some(IO_TIMEOUT))?;
    Ok(connection)
}

pub(crate) fn client(endpoint: &RedisEndpoint) -> redis::RedisResult<redis::Client> {
    let mut settings = RedisConnectionInfo::default().set_skip_set_lib_name();
    if let Some(password) = endpoint
        .password
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        settings = settings.set_password(password);
    }
    let info = (endpoint.host.clone(), endpoint.port)
        .into_connection_info()?
        .set_redis_settings(settings);
    redis::Client::open(info)
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::{
        error::Error,
        io::{BufRead, BufReader},
        net::TcpStream,
    };

    pub(crate) fn read_command(
        stream: &mut TcpStream,
    ) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
        read_command_from(&mut BufReader::new(stream))
    }

    pub(crate) fn read_command_from<R: BufRead>(
        reader: &mut R,
    ) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let count = line
            .strip_prefix('*')
            .and_then(|value| value.trim_end().parse::<usize>().ok())
            .ok_or("invalid Redis array header")?;
        let mut parts = Vec::with_capacity(count);
        for _ in 0..count {
            line.clear();
            reader.read_line(&mut line)?;
            let length = line
                .strip_prefix('$')
                .and_then(|value| value.trim_end().parse::<usize>().ok())
                .ok_or("invalid Redis bulk-string header")?;
            let mut bytes = vec![0; length];
            reader.read_exact(&mut bytes)?;
            let mut terminator = [0; 2];
            reader.read_exact(&mut terminator)?;
            if terminator != *b"\r\n" {
                return Err("invalid Redis bulk-string terminator".into());
            }
            parts.push(String::from_utf8(bytes)?);
        }
        Ok(parts)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::io::Write;
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use super::test_support::{read_command, read_command_from};
    use super::{RedisEndpoint, RedisStringCommands, client, pool};

    type TestResult = Result<(), Box<dyn Error + Send + Sync>>;

    fn endpoint(port: u16, password: Option<&str>) -> RedisEndpoint {
        RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            password: password.map(str::to_owned),
        }
    }

    #[test]
    fn pools_are_shared_per_endpoint_and_password_until_dropped() -> TestResult {
        let first = pool(&endpoint(1, Some("synthetic-a")))?;
        let same = pool(&endpoint(1, Some("synthetic-a")))?;
        let other_password = pool(&endpoint(1, Some("synthetic-b")))?;
        assert!(Arc::ptr_eq(&first.inner, &same.inner));
        assert!(!Arc::ptr_eq(&first.inner, &other_password.inner));

        let weak = Arc::downgrade(&first.inner);
        drop((first, same));
        assert!(weak.upgrade().is_none());
        Ok(())
    }

    #[test]
    fn raw_connections_authenticate_and_speak_setex_and_get() -> TestResult {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> TestResult {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            assert_eq!(read_command(&mut stream)?, ["AUTH", "synthetic-password"]);
            stream.write_all(b"+OK\r\n")?;
            let setex = read_command(&mut stream)?;
            assert_eq!(setex, ["SETEX", "synthetic:key", "30", "value"]);
            stream.write_all(b"+OK\r\n")?;
            assert_eq!(read_command(&mut stream)?, ["GET", "synthetic:key"]);
            stream.write_all(b"$5\r\nvalue\r\n")?;
            Ok(())
        });

        let mut connection =
            client(&endpoint(port, Some("synthetic-password")))?.get_connection()?;
        connection.set_text("synthetic:key", "value", 30)?;
        assert_eq!(
            connection.get_text("synthetic:key")?.as_deref(),
            Some("value")
        );
        server
            .join()
            .ok()
            .ok_or("synthetic Redis server panicked")??;
        Ok(())
    }

    #[test]
    fn empty_passwords_skip_authentication() -> TestResult {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> TestResult {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            assert_eq!(read_command(&mut stream)?, ["GET", "synthetic:missing"]);
            stream.write_all(b"$-1\r\n")?;
            Ok(())
        });

        let mut connection = client(&endpoint(port, Some("")))?.get_connection()?;
        assert_eq!(connection.get_text("synthetic:missing")?, None);
        server
            .join()
            .ok()
            .ok_or("synthetic Redis server panicked")??;
        Ok(())
    }

    #[test]
    fn synthetic_server_parser_rejects_malformed_commands() -> TestResult {
        let valid = read_command_from(&mut "*2\r\n$3\r\nGET\r\n$1\r\nk\r\n".as_bytes())?;
        assert_eq!(valid, ["GET", "k"]);
        for (frame, error) in [
            ("GET k\r\n", "invalid Redis array header"),
            ("*1\r\n:3\r\n", "invalid Redis bulk-string header"),
            ("*1\r\n$3\r\nGETxx", "invalid Redis bulk-string terminator"),
        ] {
            let parsed = read_command_from(&mut frame.as_bytes());
            assert_eq!(
                parsed.map_err(|failure| failure.to_string()),
                Err(error.to_owned())
            );
        }
        Ok(())
    }
}
