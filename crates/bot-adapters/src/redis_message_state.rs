//! Redis I/O for summaries, compaction markers, bot metadata, and chat members.

use std::collections::HashMap;

use serde::Serialize;
use thiserror::Error;

use crate::redis_connection::{RedisEndpoint, client};

#[derive(Debug, Error)]
pub enum RedisMessageStateError {
    #[error("Redis message-state operation failed: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("message-state TTL must be non-negative")]
    InvalidTtl,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct StoredChatMember {
    pub user_id: String,
    pub payload: String,
}

pub struct RedisMessageState {
    client: redis::Client,
}

impl RedisMessageState {
    pub fn new(endpoint: &RedisEndpoint) -> Result<Self, RedisMessageStateError> {
        Ok(Self {
            client: client(endpoint)?,
        })
    }

    pub fn get_value(&self, key: &str) -> Result<Option<String>, RedisMessageStateError> {
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("GET").arg(key).query(&mut connection)?)
    }

    pub fn set_value(
        &self,
        key: &str,
        value: &str,
        ttl_seconds: i64,
    ) -> Result<(), RedisMessageStateError> {
        let ttl_seconds = valid_ttl(ttl_seconds)?;
        let mut connection = self.client.get_connection()?;
        redis::cmd("SETEX")
            .arg(key)
            .arg(ttl_seconds)
            .arg(value)
            .query::<()>(&mut connection)?;
        Ok(())
    }

    pub fn save_compaction_result(
        &self,
        summary_key: &str,
        marker_key: &str,
        summary: &str,
        marker: &str,
        ttl_seconds: i64,
    ) -> Result<(), RedisMessageStateError> {
        let ttl_seconds = valid_ttl(ttl_seconds)?;
        let mut connection = self.client.get_connection()?;
        redis::pipe()
            .atomic()
            .cmd("SETEX")
            .arg(summary_key)
            .arg(ttl_seconds)
            .arg(summary)
            .ignore()
            .cmd("SETEX")
            .arg(marker_key)
            .arg(ttl_seconds)
            .arg(marker)
            .ignore()
            .query::<()>(&mut connection)?;
        Ok(())
    }

    pub fn save_chat_member(
        &self,
        key: &str,
        user_id: &str,
        payload: &str,
        ttl_seconds: i64,
    ) -> Result<(), RedisMessageStateError> {
        let ttl_seconds = valid_ttl(ttl_seconds)?;
        let mut connection = self.client.get_connection()?;
        redis::pipe()
            .atomic()
            .cmd("HSET")
            .arg(key)
            .arg(user_id)
            .arg(payload)
            .ignore()
            .cmd("EXPIRE")
            .arg(key)
            .arg(ttl_seconds)
            .ignore()
            .query::<()>(&mut connection)?;
        Ok(())
    }

    pub fn get_chat_members(
        &self,
        key: &str,
    ) -> Result<Vec<StoredChatMember>, RedisMessageStateError> {
        let mut connection = self.client.get_connection()?;
        let members: HashMap<String, String> =
            redis::cmd("HGETALL").arg(key).query(&mut connection)?;
        let mut members = members
            .into_iter()
            .map(|(user_id, payload)| StoredChatMember { user_id, payload })
            .collect::<Vec<_>>();
        members.sort_unstable_by(|left, right| left.user_id.cmp(&right.user_id));
        Ok(members)
    }
}

fn valid_ttl(ttl_seconds: i64) -> Result<u64, RedisMessageStateError> {
    u64::try_from(ttl_seconds).map_err(|_| RedisMessageStateError::InvalidTtl)
}

#[cfg(test)]
mod tests {
    use std::{
        error::Error,
        io::{BufReader, Write},
        net::TcpListener,
        thread,
        time::Duration,
    };

    use super::{RedisMessageState, RedisMessageStateError, StoredChatMember};
    use crate::redis_connection::{
        RedisEndpoint,
        test_support::{read_command, read_command_from},
    };

    #[test]
    fn preserves_auxiliary_keys_ttls_transactions_and_member_payloads()
    -> Result<(), Box<dyn Error + Send + Sync>> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<(), Box<dyn Error + Send + Sync>> {
            let (mut get_stream, _) = listener.accept()?;
            get_stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            assert_eq!(read_command(&mut get_stream)?, ["GET", "chat_summary:1"]);
            get_stream.write_all(b"$7\r\nsummary\r\n")?;

            let (mut set_stream, _) = listener.accept()?;
            set_stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            assert_eq!(
                read_command(&mut set_stream)?,
                ["SETEX", "chat_summary:1", "300", "fresh"]
            );
            set_stream.write_all(b"+OK\r\n")?;

            let (pair_stream, _) = listener.accept()?;
            pair_stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            let mut pair_reader = BufReader::new(pair_stream);
            assert_eq!(read_command_from(&mut pair_reader)?, ["MULTI"]);
            pair_reader.get_mut().write_all(b"+OK\r\n")?;
            assert_eq!(
                read_command_from(&mut pair_reader)?,
                ["SETEX", "chat_summary:1", "300", "summary"]
            );
            pair_reader.get_mut().write_all(b"+QUEUED\r\n")?;
            assert_eq!(
                read_command_from(&mut pair_reader)?,
                ["SETEX", "chat_compacted_until:1", "300", "42"]
            );
            pair_reader.get_mut().write_all(b"+QUEUED\r\n")?;
            assert_eq!(read_command_from(&mut pair_reader)?, ["EXEC"]);
            pair_reader.get_mut().write_all(b"*2\r\n+OK\r\n+OK\r\n")?;

            let (member_stream, _) = listener.accept()?;
            member_stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            let mut member_reader = BufReader::new(member_stream);
            assert_eq!(read_command_from(&mut member_reader)?, ["MULTI"]);
            member_reader.get_mut().write_all(b"+OK\r\n")?;
            assert_eq!(
                read_command_from(&mut member_reader)?,
                ["HSET", "chat_members:1", "7", "member-json"]
            );
            member_reader.get_mut().write_all(b"+QUEUED\r\n")?;
            assert_eq!(
                read_command_from(&mut member_reader)?,
                ["EXPIRE", "chat_members:1", "300"]
            );
            member_reader.get_mut().write_all(b"+QUEUED\r\n")?;
            assert_eq!(read_command_from(&mut member_reader)?, ["EXEC"]);
            member_reader.get_mut().write_all(b"*2\r\n:1\r\n:1\r\n")?;

            let (mut members_stream, _) = listener.accept()?;
            members_stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            assert_eq!(
                read_command(&mut members_stream)?,
                ["HGETALL", "chat_members:1"]
            );
            members_stream
                .write_all(b"*4\r\n$1\r\n8\r\n$8\r\nmember-8\r\n$1\r\n7\r\n$8\r\nmember-7\r\n")?;
            Ok(())
        });
        let state = RedisMessageState::new(&RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            password: None,
        })?;

        assert_eq!(
            state.get_value("chat_summary:1")?,
            Some("summary".to_owned())
        );
        state.set_value("chat_summary:1", "fresh", 300)?;
        state.save_compaction_result(
            "chat_summary:1",
            "chat_compacted_until:1",
            "summary",
            "42",
            300,
        )?;
        state.save_chat_member("chat_members:1", "7", "member-json", 300)?;
        assert_eq!(
            state.get_chat_members("chat_members:1")?,
            [
                StoredChatMember {
                    user_id: "7".to_owned(),
                    payload: "member-7".to_owned(),
                },
                StoredChatMember {
                    user_id: "8".to_owned(),
                    payload: "member-8".to_owned(),
                },
            ]
        );

        match server.join() {
            Ok(result) => result?,
            Err(_) => return Err("synthetic Redis server panicked".into()),
        }
        Ok(())
    }

    #[test]
    fn rejects_negative_ttl_before_connecting() -> Result<(), RedisMessageStateError> {
        let state = RedisMessageState::new(&RedisEndpoint {
            host: "invalid.invalid".to_owned(),
            port: 1,
            password: None,
        })?;
        assert!(matches!(
            state.set_value("key", "value", -1),
            Err(RedisMessageStateError::InvalidTtl)
        ));
        assert!(matches!(
            state.save_chat_member("key", "user", "payload", -1),
            Err(RedisMessageStateError::InvalidTtl)
        ));
        Ok(())
    }
}
