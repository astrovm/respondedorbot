//! Redis cache adapter for Telegram chat-administrator checks.

use serde_json::{Value, json};
use thiserror::Error;

use crate::redis_connection::{RedisEndpoint, RedisStringCommands, connect};

#[derive(Debug, Error)]
pub enum RedisChatAdminError {
    #[error("Redis chat-admin cache operation failed: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("chat-admin cache TTL must be non-negative")]
    InvalidTtl,
}

#[must_use]
pub fn chat_admin_cache_key(chat_id: &str, user_id: &str) -> String {
    format!("chat_admin:{chat_id}:{user_id}")
}

pub fn get_cached_chat_admin_with<C: RedisStringCommands>(
    connection: &mut C,
    chat_id: &str,
    user_id: &str,
) -> Result<Option<bool>, RedisChatAdminError> {
    let Some(text) = connection.get_text(&chat_admin_cache_key(chat_id, user_id))? else {
        return Ok(None);
    };
    let Ok(value) = serde_json::from_str::<Value>(&text) else {
        return Ok(None);
    };
    Ok(match value {
        Value::Bool(flag) => Some(flag),
        Value::Object(object) => object.get("is_admin").and_then(Value::as_bool),
        _ => None,
    })
}

pub fn cache_chat_admin_with<C: RedisStringCommands>(
    connection: &mut C,
    chat_id: &str,
    user_id: &str,
    is_admin: bool,
    ttl_seconds: i64,
) -> Result<(), RedisChatAdminError> {
    let ttl_seconds = u64::try_from(ttl_seconds).map_err(|_| RedisChatAdminError::InvalidTtl)?;
    let value = json!({"is_admin": is_admin}).to_string();
    connection.set_text(&chat_admin_cache_key(chat_id, user_id), &value, ttl_seconds)?;
    Ok(())
}

pub fn get_cached_chat_admin(
    endpoint: &RedisEndpoint,
    chat_id: &str,
    user_id: &str,
) -> Result<Option<bool>, RedisChatAdminError> {
    let mut connection = connect(endpoint)?;
    get_cached_chat_admin_with(&mut connection, chat_id, user_id)
}

pub fn cache_chat_admin(
    endpoint: &RedisEndpoint,
    chat_id: &str,
    user_id: &str,
    is_admin: bool,
    ttl_seconds: i64,
) -> Result<(), RedisChatAdminError> {
    let mut connection = connect(endpoint)?;
    cache_chat_admin_with(&mut connection, chat_id, user_id, is_admin, ttl_seconds)
}

#[cfg(test)]
mod tests {
    use std::{error::Error, io::Write, net::TcpListener, thread, time::Duration};

    use super::{
        RedisChatAdminError, cache_chat_admin, cache_chat_admin_with, chat_admin_cache_key,
        get_cached_chat_admin, get_cached_chat_admin_with,
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
    fn reads_legacy_boolean_and_object_payloads() -> Result<(), RedisChatAdminError> {
        assert_eq!(chat_admin_cache_key("chat", "42"), "chat_admin:chat:42");
        for (value, expected) in [
            ("true", Some(true)),
            ("false", Some(false)),
            (r#"{"is_admin":true}"#, Some(true)),
            (r#"{"is_admin":false}"#, Some(false)),
            (r#"{"is_admin":"yes"}"#, None),
            ("invalid", None),
        ] {
            let mut commands = FakeCommands {
                value: Some(value.to_owned()),
                ..FakeCommands::default()
            };
            assert_eq!(
                get_cached_chat_admin_with(&mut commands, "chat", "42")?,
                expected
            );
            assert_eq!(commands.gets, ["chat_admin:chat:42"]);
        }
        Ok(())
    }

    #[test]
    fn writes_python_compatible_versionless_json_and_ttl() -> Result<(), RedisChatAdminError> {
        let mut commands = FakeCommands::default();
        cache_chat_admin_with(&mut commands, "chat", "42", true, 300)?;
        assert_eq!(
            commands.sets,
            [(
                "chat_admin:chat:42".to_owned(),
                r#"{"is_admin":true}"#.to_owned(),
                300
            )]
        );
        Ok(())
    }

    #[test]
    fn rejects_negative_ttl_before_writing() {
        let mut commands = FakeCommands::default();
        assert!(matches!(
            cache_chat_admin_with(&mut commands, "chat", "42", false, -1),
            Err(RedisChatAdminError::InvalidTtl)
        ));
        assert!(commands.sets.is_empty());
    }

    #[test]
    fn public_adapter_speaks_compatible_get_and_setex() -> Result<(), Box<dyn Error + Send + Sync>>
    {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<(), Box<dyn Error + Send + Sync>> {
            let (mut get_stream, _) = listener.accept()?;
            get_stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            assert_eq!(
                read_command(&mut get_stream)?,
                ["GET", "chat_admin:chat:42"]
            );
            get_stream.write_all(b"$17\r\n{\"is_admin\":true}\r\n")?;

            let (mut set_stream, _) = listener.accept()?;
            set_stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            assert_eq!(
                read_command(&mut set_stream)?,
                [
                    "SETEX",
                    "chat_admin:chat:42",
                    "300",
                    r#"{"is_admin":false}"#,
                ]
            );
            set_stream.write_all(b"+OK\r\n")?;
            Ok(())
        });
        let endpoint = RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            password: None,
        };

        assert_eq!(get_cached_chat_admin(&endpoint, "chat", "42")?, Some(true));
        cache_chat_admin(&endpoint, "chat", "42", false, 300)?;

        match server.join() {
            Ok(result) => result?,
            Err(_) => return Err("synthetic Redis server panicked".into()),
        }
        Ok(())
    }
}
