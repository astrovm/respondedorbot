//! Single-owner Redis adapter for expensive media results.

use redis::{Commands, IntoConnectionInfo, RedisConnectionInfo};
use thiserror::Error;

#[derive(Clone)]
pub struct RedisEndpoint {
    pub host: String,
    pub port: u16,
    pub password: Option<String>,
}

#[derive(Debug, Error)]
pub enum RedisMediaCacheError {
    #[error("Redis media-cache operation failed: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("media-cache TTL must be non-negative")]
    InvalidTtl,
}

pub trait MediaCacheCommands {
    fn get_text(&mut self, key: &str) -> redis::RedisResult<Option<String>>;
    fn set_text(&mut self, key: &str, value: &str, ttl_seconds: u64) -> redis::RedisResult<()>;
}

impl MediaCacheCommands for redis::Connection {
    fn get_text(&mut self, key: &str) -> redis::RedisResult<Option<String>> {
        self.get(key)
    }

    fn set_text(&mut self, key: &str, value: &str, ttl_seconds: u64) -> redis::RedisResult<()> {
        self.set_ex(key, value, ttl_seconds)
    }
}

#[must_use]
pub fn media_cache_key(prefix: &str, file_id: &str) -> String {
    format!("{prefix}:{file_id}")
}

pub fn get_cached_media_with<C: MediaCacheCommands>(
    connection: &mut C,
    prefix: &str,
    file_id: &str,
) -> Result<Option<String>, RedisMediaCacheError> {
    Ok(connection.get_text(&media_cache_key(prefix, file_id))?)
}

pub fn cache_media_with<C: MediaCacheCommands>(
    connection: &mut C,
    prefix: &str,
    file_id: &str,
    text: &str,
    ttl_seconds: i64,
) -> Result<(), RedisMediaCacheError> {
    let ttl_seconds = u64::try_from(ttl_seconds).map_err(|_| RedisMediaCacheError::InvalidTtl)?;
    connection.set_text(&media_cache_key(prefix, file_id), text, ttl_seconds)?;
    Ok(())
}

pub fn get_cached_media(
    endpoint: &RedisEndpoint,
    prefix: &str,
    file_id: &str,
) -> Result<Option<String>, RedisMediaCacheError> {
    let mut connection = connect(endpoint)?;
    get_cached_media_with(&mut connection, prefix, file_id)
}

pub fn cache_media(
    endpoint: &RedisEndpoint,
    prefix: &str,
    file_id: &str,
    text: &str,
    ttl_seconds: i64,
) -> Result<(), RedisMediaCacheError> {
    let mut connection = connect(endpoint)?;
    cache_media_with(&mut connection, prefix, file_id, text, ttl_seconds)
}

fn connect(endpoint: &RedisEndpoint) -> Result<redis::Connection, RedisMediaCacheError> {
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
    Ok(redis::Client::open(info)?.get_connection()?)
}

#[cfg(test)]
mod tests {
    use std::{
        error::Error,
        io::{BufRead, BufReader, Read, Write},
        net::{TcpListener, TcpStream},
        thread,
        time::Duration,
    };

    use super::{
        MediaCacheCommands, RedisEndpoint, RedisMediaCacheError, cache_media, cache_media_with,
        get_cached_media, get_cached_media_with, media_cache_key,
    };

    #[derive(Default)]
    struct FakeCommands {
        value: Option<String>,
        gets: Vec<String>,
        sets: Vec<(String, String, u64)>,
    }

    impl MediaCacheCommands for FakeCommands {
        fn get_text(&mut self, key: &str) -> redis::RedisResult<Option<String>> {
            self.gets.push(key.to_owned());
            Ok(self.value.clone())
        }

        fn set_text(&mut self, key: &str, value: &str, ttl_seconds: u64) -> redis::RedisResult<()> {
            self.sets
                .push((key.to_owned(), value.to_owned(), ttl_seconds));
            Ok(())
        }
    }

    #[test]
    fn preserves_media_keys_hits_misses_and_ttls() -> Result<(), RedisMediaCacheError> {
        assert_eq!(
            media_cache_key("audio_transcription", "file"),
            "audio_transcription:file"
        );
        let mut commands = FakeCommands {
            value: Some("cached".to_owned()),
            ..FakeCommands::default()
        };
        assert_eq!(
            get_cached_media_with(&mut commands, "audio_transcription", "file")?,
            Some("cached".to_owned())
        );
        cache_media_with(
            &mut commands,
            "image_description",
            "image",
            "description",
            3_600,
        )?;
        assert_eq!(commands.gets, ["audio_transcription:file"]);
        assert_eq!(
            commands.sets,
            [(
                "image_description:image".to_owned(),
                "description".to_owned(),
                3_600
            )]
        );
        Ok(())
    }

    #[test]
    fn rejects_negative_ttl_before_any_redis_write() {
        let mut commands = FakeCommands::default();
        let result = cache_media_with(&mut commands, "audio", "file", "text", -1);
        assert!(matches!(result, Err(RedisMediaCacheError::InvalidTtl)));
        assert!(commands.sets.is_empty());
    }

    #[test]
    fn public_adapter_speaks_redis_get_and_setex() -> Result<(), Box<dyn Error + Send + Sync>> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<(), Box<dyn Error + Send + Sync>> {
            let (mut get_stream, _) = listener.accept()?;
            get_stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            assert_eq!(read_command(&mut get_stream)?, ["GET", "media:file"]);
            get_stream.write_all(b"$6\r\ncached\r\n")?;

            let (mut set_stream, _) = listener.accept()?;
            set_stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            assert_eq!(
                read_command(&mut set_stream)?,
                ["SETEX", "media:file", "60", "fresh"]
            );
            set_stream.write_all(b"+OK\r\n")?;
            Ok(())
        });
        let endpoint = RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            password: None,
        };

        assert_eq!(
            get_cached_media(&endpoint, "media", "file")?,
            Some("cached".to_owned())
        );
        cache_media(&endpoint, "media", "file", "fresh", 60)?;

        match server.join() {
            Ok(result) => result?,
            Err(_) => return Err("synthetic Redis server panicked".into()),
        }
        Ok(())
    }

    fn read_command(stream: &mut TcpStream) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
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
