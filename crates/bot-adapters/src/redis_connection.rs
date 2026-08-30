//! Shared synchronous Redis connection boundary for temporary migration adapters.

use redis::{Commands, IntoConnectionInfo, RedisConnectionInfo};

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

pub(crate) fn connect(endpoint: &RedisEndpoint) -> redis::RedisResult<redis::Connection> {
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
    redis::Client::open(info)?.get_connection()
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::{
        error::Error,
        io::{BufRead, BufReader, Read},
        net::TcpStream,
    };

    pub(crate) fn read_command(
        stream: &mut TcpStream,
    ) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
        let mut reader = BufReader::new(stream);
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
