use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::message_utils::{truncate_text, ChatHistoryEntry};

#[cfg(not(target_arch = "wasm32"))]
type RedisClient = redis::Client;

#[derive(Clone)]
pub struct Storage {
    backend: StorageBackend,
}

#[derive(Clone)]
enum StorageBackend {
    #[cfg(not(target_arch = "wasm32"))]
    Redis(RedisClient),
    #[cfg(target_arch = "wasm32")]
    Kv(worker::kv::KvStore),
}

#[derive(Debug, Serialize, Deserialize)]
struct StoredCounter {
    value: i64,
    expires_at: i64,
}

#[derive(Debug, Serialize, Deserialize)]
struct StoredMessage {
    id: String,
    text: String,
    timestamp: i64,
}

const CHAT_HISTORY_LIMIT: usize = 32;
const BOT_MESSAGE_META_PREFIX: &str = "bot_message_meta:";

impl Storage {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn create_redis_client(
        host: &str,
        port: u16,
        password: Option<&str>,
    ) -> redis::RedisResult<RedisClient> {
        let url = if let Some(pw) = password {
            format!("redis://:{}@{}:{}/", pw, host, port)
        } else {
            format!("redis://{}:{}/", host, port)
        };
        redis::Client::open(url)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_redis_client(client: RedisClient) -> Self {
        Self {
            backend: StorageBackend::Redis(client),
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_kv(kv: worker::kv::KvStore) -> Self {
        Self {
            backend: StorageBackend::Kv(kv),
        }
    }

    pub async fn get_string(&self, key: &str) -> Option<String> {
        match &self.backend {
            #[cfg(not(target_arch = "wasm32"))]
            StorageBackend::Redis(client) => {
                let mut conn = client.get_multiplexed_async_connection().await.ok()?;
                redis::AsyncCommands::get(&mut conn, key).await.ok()
            }
            #[cfg(target_arch = "wasm32")]
            StorageBackend::Kv(kv) => kv.get(key).text().await.ok().flatten(),
        }
    }

    pub async fn set_string(&self, key: &str, value: &str) -> bool {
        match &self.backend {
            #[cfg(not(target_arch = "wasm32"))]
            StorageBackend::Redis(client) => {
                let mut conn = match client.get_multiplexed_async_connection().await {
                    Ok(conn) => conn,
                    Err(_) => return false,
                };
                redis::AsyncCommands::set::<_, _, bool>(&mut conn, key, value)
                    .await
                    .unwrap_or(false)
            }
            #[cfg(target_arch = "wasm32")]
            StorageBackend::Kv(kv) => kv
                .put(key, value)
                .and_then(|builder| builder.execute())
                .await
                .is_ok(),
        }
    }

    pub async fn set_string_with_ttl(&self, key: &str, ttl_seconds: u64, value: &str) -> bool {
        match &self.backend {
            #[cfg(not(target_arch = "wasm32"))]
            StorageBackend::Redis(client) => {
                let mut conn = match client.get_multiplexed_async_connection().await {
                    Ok(conn) => conn,
                    Err(_) => return false,
                };
                redis::AsyncCommands::set_ex::<_, _, bool>(&mut conn, key, value, ttl_seconds)
                    .await
                    .unwrap_or(false)
            }
            #[cfg(target_arch = "wasm32")]
            StorageBackend::Kv(kv) => kv
                .put(key, value)
                .and_then(|builder| Ok(builder.expiration_ttl(ttl_seconds)))
                .and_then(|builder| builder.execute())
                .await
                .is_ok(),
        }
    }

    pub async fn set_json(
        &self,
        key: &str,
        value: &serde_json::Value,
        ttl_seconds: Option<u64>,
    ) -> bool {
        let payload = value.to_string();
        match ttl_seconds {
            Some(ttl) => self.set_string_with_ttl(key, ttl, &payload).await,
            None => self.set_string(key, &payload).await,
        }
    }

    pub async fn get_json(&self, key: &str) -> Option<serde_json::Value> {
        let raw = self.get_string(key).await?;
        serde_json::from_str::<serde_json::Value>(&raw).ok()
    }

    pub async fn incr_with_ttl(&self, key: &str, ttl_seconds: u64) -> i64 {
        match &self.backend {
            #[cfg(not(target_arch = "wasm32"))]
            StorageBackend::Redis(client) => {
                let mut conn = match client.get_multiplexed_async_connection().await {
                    Ok(conn) => conn,
                    Err(_) => return 0,
                };
                let count: i64 = match redis::AsyncCommands::incr(&mut conn, key, 1).await {
                    Ok(value) => value,
                    Err(_) => return 0,
                };
                if count == 1 {
                    let _: () = redis::AsyncCommands::expire(&mut conn, key, ttl_seconds as i64)
                        .await
                        .unwrap_or(());
                }
                count
            }
            #[cfg(target_arch = "wasm32")]
            StorageBackend::Kv(kv) => {
                let now = current_timestamp();
                let current = self
                    .get_json(key)
                    .await
                    .and_then(|raw| serde_json::from_value::<StoredCounter>(raw).ok());

                let mut value = 1;
                let mut expires_at = now + ttl_seconds as i64;
                if let Some(counter) = current {
                    if counter.expires_at >= now {
                        value = counter.value + 1;
                        expires_at = counter.expires_at;
                    }
                }

                let remaining_ttl = expires_at.saturating_sub(now).max(1) as u64;
                let payload = serde_json::json!({
                    "value": value,
                    "expires_at": expires_at,
                });
                let _ = kv
                    .put(key, payload.to_string())
                    .and_then(|builder| Ok(builder.expiration_ttl(remaining_ttl)))
                    .and_then(|builder| builder.execute())
                    .await;
                value
            }
        }
    }

    pub async fn append_chat_history(&self, chat_id: i64, message_id: &str, text: &str) {
        let key = format!("chat_history:{chat_id}");
        let mut history: Vec<StoredMessage> = self
            .get_json(&key)
            .await
            .and_then(|value| serde_json::from_value(value).ok())
            .unwrap_or_default();

        if history.iter().any(|entry| entry.id == message_id) {
            return;
        }

        let entry = StoredMessage {
            id: message_id.to_string(),
            text: truncate_text(text, 512),
            timestamp: current_timestamp(),
        };

        history.insert(0, entry);
        history.truncate(CHAT_HISTORY_LIMIT);

        let _ = self.set_json(&key, &serde_json::json!(history), None).await;
    }

    pub async fn load_chat_history(
        &self,
        chat_id: i64,
        max_messages: usize,
    ) -> Vec<ChatHistoryEntry> {
        let key = format!("chat_history:{chat_id}");
        let history: Vec<StoredMessage> = self
            .get_json(&key)
            .await
            .and_then(|value| serde_json::from_value(value).ok())
            .unwrap_or_default();

        let mut out = Vec::new();
        for entry in history.into_iter().take(max_messages) {
            if entry.text.trim().is_empty() {
                continue;
            }
            let role = if entry.id.starts_with("bot_") {
                "assistant".to_string()
            } else {
                "user".to_string()
            };
            out.push(ChatHistoryEntry {
                role,
                text: entry.text,
            });
        }
        out.reverse();
        out
    }

    pub async fn save_bot_message_metadata(
        &self,
        chat_id: i64,
        message_id: i64,
        metadata: &serde_json::Value,
        ttl_seconds: u64,
    ) {
        let key = bot_message_meta_key(chat_id, message_id);
        let _ = self.set_json(&key, metadata, Some(ttl_seconds)).await;
    }

    pub async fn load_bot_message_metadata(
        &self,
        chat_id: i64,
        message_id: i64,
    ) -> Option<serde_json::Value> {
        let key = bot_message_meta_key(chat_id, message_id);
        self.get_json(&key).await
    }
}

fn bot_message_meta_key(chat_id: i64, message_id: i64) -> String {
    format!("{BOT_MESSAGE_META_PREFIX}{chat_id}:{message_id}")
}

fn current_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}
