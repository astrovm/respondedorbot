//! Redis I/O for summaries, compaction markers, bot metadata, and chat members.

use std::{
    collections::{BTreeMap, HashMap},
    sync::atomic::{AtomicBool, Ordering},
};

use serde::Serialize;
use thiserror::Error;

use bot_core::message_state::{MessageWritePlan, escape_search_tag, escape_search_text};

use crate::redis_connection::{RedisEndpoint, client};

const CHAT_SEARCH_INDEX: &str = "idx:chat_messages";

const SAVE_MESSAGE_SCRIPT: &str = r#"
local legacy_type = redis.call('TYPE', KEYS[3]).ok
if legacy_type == 'set' and redis.call('SISMEMBER', KEYS[3], ARGV[1]) == 1 then
    return 0
end
if redis.call('ZSCORE', KEYS[2], ARGV[1]) then
    return 0
end

local sequence = redis.call('INCR', KEYS[4])
redis.call('ZADD', KEYS[2], sequence, ARGV[1])
redis.call('LPUSH', KEYS[1], ARGV[2])
redis.call('LTRIM', KEYS[1], 0, tonumber(ARGV[4]) - 1)

local indexed_count = redis.call('ZCARD', KEYS[2])
local max_messages = tonumber(ARGV[4])
if indexed_count > max_messages then
    redis.call('ZREMRANGEBYRANK', KEYS[2], 0, indexed_count - max_messages - 1)
end

redis.call('EXPIRE', KEYS[1], ARGV[3])
redis.call('EXPIRE', KEYS[2], ARGV[3])
redis.call('EXPIRE', KEYS[4], ARGV[3])
redis.call(
    'HSET',
    KEYS[5],
    'chat_id', ARGV[5],
    'message_id', ARGV[1],
    'role', ARGV[6],
    'user_id', ARGV[7],
    'username', ARGV[8],
    'text', ARGV[9],
    'timestamp', ARGV[10],
    'reply_to_message_id', ARGV[11],
    'mentions_bot', ARGV[12]
)
redis.call('EXPIRE', KEYS[5], ARGV[3])
return 1
"#;

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

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct SearchRow {
    pub key: String,
    pub fields: BTreeMap<String, String>,
}

pub struct RedisMessageState {
    client: redis::Client,
    search_index_ready: AtomicBool,
}

impl RedisMessageState {
    pub fn new(endpoint: &RedisEndpoint) -> Result<Self, RedisMessageStateError> {
        Ok(Self {
            client: client(endpoint)?,
            search_index_ready: AtomicBool::new(false),
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

    pub fn save_message(
        &self,
        plan: &MessageWritePlan,
        ttl_seconds: i64,
        max_messages: usize,
    ) -> Result<bool, RedisMessageStateError> {
        let ttl_seconds = valid_ttl(ttl_seconds)?;
        let _ = self.ensure_search_index();
        let mut connection = self.client.get_connection()?;
        let stored: i64 = redis::cmd("EVAL")
            .arg(SAVE_MESSAGE_SCRIPT)
            .arg(5)
            .arg(&plan.keys.history)
            .arg(&plan.keys.order)
            .arg(&plan.keys.legacy_ids)
            .arg(&plan.keys.sequence)
            .arg(&plan.keys.search_document)
            .arg(&plan.message_id)
            .arg(&plan.history_entry)
            .arg(ttl_seconds)
            .arg(max_messages)
            .arg(&plan.chat_id)
            .arg(&plan.role)
            .arg(&plan.user_id)
            .arg(&plan.username)
            .arg(&plan.text)
            .arg(plan.timestamp)
            .arg(&plan.reply_to_message_id)
            .arg(&plan.mentions_bot)
            .query(&mut connection)?;
        Ok(stored > 0)
    }

    pub fn get_history_entries(
        &self,
        chat_id: &str,
        max_messages: i64,
    ) -> Result<Vec<String>, RedisMessageStateError> {
        let mut connection = self.client.get_connection()?;
        Ok(redis::cmd("LRANGE")
            .arg(format!("chat_history:{chat_id}"))
            .arg(0)
            .arg(max_messages.saturating_sub(1))
            .query(&mut connection)?)
    }

    pub fn fetch_messages(
        &self,
        chat_id: &str,
        limit: usize,
    ) -> Result<Vec<SearchRow>, RedisMessageStateError> {
        let query = format!("@chat_id:{{{}}}", escape_search_tag(chat_id));
        self.search(&query, limit)
    }

    pub fn search_messages(
        &self,
        chat_id: &str,
        query_text: &str,
        limit: usize,
    ) -> Result<Vec<SearchRow>, RedisMessageStateError> {
        let query = format!(
            "@chat_id:{{{}}} {}",
            escape_search_tag(chat_id),
            escape_search_text(query_text)
        );
        self.search(&query, limit)
    }

    fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchRow>, RedisMessageStateError> {
        self.ensure_search_index()?;
        let mut connection = self.client.get_connection()?;
        let value: redis::Value = redis::cmd("FT.SEARCH")
            .arg(CHAT_SEARCH_INDEX)
            .arg(query)
            .arg("DIALECT")
            .arg(2)
            .arg("SORTBY")
            .arg("timestamp")
            .arg("DESC")
            .arg("LIMIT")
            .arg(0)
            .arg(limit)
            .query(&mut connection)?;
        Ok(parse_search_rows(value))
    }

    fn ensure_search_index(&self) -> Result<(), RedisMessageStateError> {
        if self.search_index_ready.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut connection = self.client.get_connection()?;
        let result = redis::cmd("FT.CREATE")
            .arg(CHAT_SEARCH_INDEX)
            .arg("ON")
            .arg("HASH")
            .arg("PREFIX")
            .arg(1)
            .arg("chatmsg:")
            .arg("SCHEMA")
            .arg("chat_id")
            .arg("TAG")
            .arg("role")
            .arg("TAG")
            .arg("user_id")
            .arg("TAG")
            .arg("reply_to_message_id")
            .arg("TAG")
            .arg("mentions_bot")
            .arg("TAG")
            .arg("username")
            .arg("TEXT")
            .arg("text")
            .arg("TEXT")
            .arg("timestamp")
            .arg("NUMERIC")
            .arg("SORTABLE")
            .query::<()>(&mut connection);
        if let Err(error) = result {
            let message = error.to_string().to_ascii_lowercase();
            if !message.contains("already exists") {
                return Err(error.into());
            }
        }
        self.search_index_ready.store(true, Ordering::Release);
        Ok(())
    }
}

fn valid_ttl(ttl_seconds: i64) -> Result<u64, RedisMessageStateError> {
    u64::try_from(ttl_seconds).map_err(|_| RedisMessageStateError::InvalidTtl)
}

fn parse_search_rows(value: redis::Value) -> Vec<SearchRow> {
    let redis::Value::Array(values) = value else {
        return Vec::new();
    };
    let values = values.into_iter().skip(1).collect::<Vec<_>>();
    let (pairs, _) = values.as_chunks::<2>();
    pairs
        .iter()
        .filter_map(|pair| {
            let key = value_text(&pair[0])?;
            let redis::Value::Array(fields) = &pair[1] else {
                return None;
            };
            let (fields, _) = fields.as_chunks::<2>();
            let fields = fields
                .iter()
                .filter_map(|field| Some((value_text(&field[0])?, value_text(&field[1])?)))
                .collect();
            Some(SearchRow { key, fields })
        })
        .collect()
}

fn value_text(value: &redis::Value) -> Option<String> {
    match value {
        redis::Value::BulkString(bytes) => Some(String::from_utf8_lossy(bytes).into_owned()),
        redis::Value::SimpleString(text) => Some(text.clone()),
        redis::Value::Int(number) => Some(number.to_string()),
        _ => None,
    }
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
    use bot_core::message_state::prepare_message_write;

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

    #[test]
    fn writes_and_reads_history_and_decodes_redisearch_rows()
    -> Result<(), Box<dyn Error + Send + Sync>> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<(), Box<dyn Error + Send + Sync>> {
            let (mut index_stream, _) = listener.accept()?;
            index_stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            let create = read_command(&mut index_stream)?;
            assert_eq!(create.first().map(String::as_str), Some("FT.CREATE"));
            index_stream.write_all(b"-Index already exists\r\n")?;

            let (mut write_stream, _) = listener.accept()?;
            write_stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            let write = read_command(&mut write_stream)?;
            assert_eq!(write.first().map(String::as_str), Some("EVAL"));
            assert!(write.iter().any(|value| value == "chat_history:1"));
            assert!(write.iter().any(|value| value == "chatmsg:1:1"));
            write_stream.write_all(b":1\r\n")?;

            let (mut history_stream, _) = listener.accept()?;
            history_stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            assert_eq!(
                read_command(&mut history_stream)?,
                ["LRANGE", "chat_history:1", "0", "1"]
            );
            history_stream.write_all(b"*2\r\n$3\r\none\r\n$3\r\ntwo\r\n")?;

            let search_response = b"*3\r\n:1\r\n$11\r\nchatmsg:1:1\r\n*6\r\n$10\r\nmessage_id\r\n$1\r\n1\r\n$4\r\ntext\r\n$5\r\nhello\r\n$9\r\ntimestamp\r\n$2\r\n10\r\n";
            let (mut fetch_stream, _) = listener.accept()?;
            fetch_stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            let fetch = read_command(&mut fetch_stream)?;
            assert_eq!(fetch.first().map(String::as_str), Some("FT.SEARCH"));
            assert_eq!(fetch.get(2).map(String::as_str), Some("@chat_id:{1}"));
            fetch_stream.write_all(search_response)?;

            let (mut search_stream, _) = listener.accept()?;
            search_stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            let search = read_command(&mut search_stream)?;
            assert_eq!(search.first().map(String::as_str), Some("FT.SEARCH"));
            assert_eq!(
                search.get(2).map(String::as_str),
                Some("@chat_id:{1} wallet")
            );
            search_stream.write_all(search_response)?;
            Ok(())
        });
        let state = RedisMessageState::new(&RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            password: None,
        })?;
        let plan = prepare_message_write(
            "1",
            "1",
            "hello",
            10,
            None,
            Some("7"),
            Some("user"),
            None,
            false,
        )?;

        assert!(state.save_message(&plan, 300, 400)?);
        assert_eq!(state.get_history_entries("1", 2)?, ["one", "two"]);
        let fetched = state.fetch_messages("1", 20)?;
        assert_eq!(fetched[0].key, "chatmsg:1:1");
        assert_eq!(fetched[0].fields["message_id"], "1");
        assert_eq!(fetched[0].fields["text"], "hello");
        let searched = state.search_messages("1", "wallet", 10)?;
        assert_eq!(searched, fetched);

        match server.join() {
            Ok(result) => result?,
            Err(_) => return Err("synthetic Redis server panicked".into()),
        }
        Ok(())
    }
}
