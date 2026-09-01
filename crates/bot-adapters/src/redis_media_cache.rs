//! Single-owner Redis adapter for expensive media results.

use thiserror::Error;

use crate::redis_connection::{RedisEndpoint, RedisStringCommands, connect};

#[derive(Debug, Error)]
pub enum RedisMediaCacheError {
    #[error("Redis media-cache operation failed: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("media-cache TTL must be non-negative")]
    InvalidTtl,
}

#[must_use]
pub fn media_cache_key(prefix: &str, file_id: &str) -> String {
    format!("{prefix}:{file_id}")
}

pub fn get_cached_media_with<C: RedisStringCommands>(
    connection: &mut C,
    prefix: &str,
    file_id: &str,
) -> Result<Option<String>, RedisMediaCacheError> {
    Ok(connection.get_text(&media_cache_key(prefix, file_id))?)
}

pub fn cache_media_with<C: RedisStringCommands>(
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

#[cfg(test)]
mod tests {
    use std::{error::Error, io::Write, net::TcpListener, thread, time::Duration};

    use super::{
        RedisMediaCacheError, cache_media, cache_media_with, get_cached_media,
        get_cached_media_with, media_cache_key,
    };
    use crate::redis_connection::{RedisEndpoint, RedisStringCommands, test_support::read_command};

    #[derive(Default)]
    struct FakeCommands {
        value: Option<String>,
        gets: Vec<String>,
        sets: Vec<(String, String, u64)>,
    }

    impl RedisStringCommands for FakeCommands {
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
}
