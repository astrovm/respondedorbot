//! Shared synchronous Redis connection boundary.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use redis::{Commands, ConnectionLike, IntoConnectionInfo, RedisConnectionInfo};

const MAX_IDLE_CONNECTIONS: usize = 16;

struct RedisPoolInner {
    client: redis::Client,
    idle: Mutex<Vec<redis::Connection>>,
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
            .and_then(|mut idle| idle.pop())
            .map_or_else(|| self.inner.client.get_connection(), Ok)?;
        Ok(RedisPooledConnection {
            connection: Some(connection),
            pool: self.inner.clone(),
        })
    }
}

impl Deref for RedisPooledConnection {
    type Target = redis::Connection;

    fn deref(&self) -> &Self::Target {
        self.connection
            .as_ref()
            .unwrap_or_else(|| unreachable!("pooled Redis connection is present until drop"))
    }
}

impl DerefMut for RedisPooledConnection {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.connection
            .as_mut()
            .unwrap_or_else(|| unreachable!("pooled Redis connection is present until drop"))
    }
}

impl Drop for RedisPooledConnection {
    fn drop(&mut self) {
        let Some(connection) = self.connection.take() else {
            return;
        };
        if connection.is_open()
            && let Ok(mut idle) = self.pool.idle.lock()
            && idle.len() < self.pool.max_idle_connections
        {
            idle.push(connection);
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
    if let Ok(mut pools) = pools.lock() {
        if let Some(inner) = pools.get(&key).and_then(Weak::upgrade) {
            return Ok(RedisPool { inner });
        }
        let inner = Arc::new(RedisPoolInner {
            client: client(endpoint)?,
            idle: Mutex::new(Vec::new()),
            max_idle_connections: MAX_IDLE_CONNECTIONS,
        });
        pools.insert(key, Arc::downgrade(&inner));
        return Ok(RedisPool { inner });
    }
    Ok(RedisPool {
        inner: Arc::new(RedisPoolInner {
            client: client(endpoint)?,
            idle: Mutex::new(Vec::new()),
            max_idle_connections: MAX_IDLE_CONNECTIONS,
        }),
    })
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
