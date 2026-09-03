//! PostgreSQL chat configuration repository.

use bot_core::chat_config::{ChatConfig, ChatConfigError};
use postgres::Client;
use serde_json::{Map, Value};
use thiserror::Error;

use crate::postgres_pool::{PooledPostgresClient, PostgresPool, PostgresPoolError};

const CHAT_CONFIG_SCHEMA_ADVISORY_LOCK_KEY: i64 = 48_610_006;

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS chat_configs (
    chat_id TEXT PRIMARY KEY,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
";

#[derive(Debug, Error)]
pub enum ChatConfigRepositoryError {
    #[error("could not initialize PostgreSQL TLS: {0}")]
    Tls(#[from] native_tls::Error),
    #[error("PostgreSQL chat configuration operation failed: {0}")]
    Postgres(#[from] postgres::Error),
    #[error(transparent)]
    InvalidConfig(#[from] ChatConfigError),
    #[error("could not encode chat configuration: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error(transparent)]
    Pool(#[from] PostgresPoolError),
}

pub struct ChatConfigRepository {
    pool: PostgresPool,
}

impl ChatConfigRepository {
    #[must_use]
    pub fn new(database_url: &str) -> Self {
        Self {
            pool: PostgresPool::shared(database_url),
        }
    }

    pub fn ensure_schema(&self) -> Result<(), ChatConfigRepositoryError> {
        let mut client = self.connect()?;
        ensure_schema(&mut client)
    }

    pub fn get(&self, chat_id: &str) -> Result<Option<ChatConfig>, ChatConfigRepositoryError> {
        let mut client = self.connect()?;
        ensure_schema(&mut client)?;
        let Some(row) = client.query_opt(
            "SELECT config FROM chat_configs WHERE chat_id = $1",
            &[&chat_id],
        )?
        else {
            return Ok(None);
        };
        let value: Value = row.get(0);
        Ok(Some(ChatConfig::from_json(&value)?))
    }

    pub fn set(
        &self,
        chat_id: &str,
        config: &ChatConfig,
    ) -> Result<ChatConfig, ChatConfigRepositoryError> {
        let value = serde_json::to_value(config)?;
        self.merge_value(chat_id, &value)?;
        Ok(config.clone())
    }

    pub fn set_changed(
        &self,
        chat_id: &str,
        previous: &ChatConfig,
        config: &ChatConfig,
    ) -> Result<ChatConfig, ChatConfigRepositoryError> {
        let value = changed_fields(previous, config)?;
        if !value.is_empty() {
            self.merge_value(chat_id, &Value::Object(value))?;
        }
        Ok(config.clone())
    }

    fn merge_value(&self, chat_id: &str, value: &Value) -> Result<(), ChatConfigRepositoryError> {
        let mut client = self.connect()?;
        ensure_schema(&mut client)?;
        client.execute(
            "INSERT INTO chat_configs (chat_id, config) VALUES ($1, $2) \
             ON CONFLICT (chat_id) DO UPDATE SET \
             config = chat_configs.config || EXCLUDED.config, updated_at = NOW()",
            &[&chat_id, value],
        )?;
        Ok(())
    }

    fn connect(&self) -> Result<PooledPostgresClient, ChatConfigRepositoryError> {
        Ok(self.pool.get()?)
    }
}

fn changed_fields(
    previous: &ChatConfig,
    config: &ChatConfig,
) -> Result<Map<String, Value>, serde_json::Error> {
    let previous = serde_json::to_value(previous)?;
    let config = serde_json::to_value(config)?;
    let previous = previous.as_object();
    Ok(config.as_object().map_or_else(Map::new, |config| {
        config
            .iter()
            .filter(|(key, value)| previous.and_then(|fields| fields.get(*key)) != Some(*value))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }))
}

fn ensure_schema(client: &mut Client) -> Result<(), ChatConfigRepositoryError> {
    let mut transaction = client.transaction()?;
    transaction.query_one(
        "SELECT pg_advisory_xact_lock($1)",
        &[&CHAT_CONFIG_SCHEMA_ADVISORY_LOCK_KEY],
    )?;
    transaction.batch_execute(SCHEMA_SQL)?;
    transaction.commit()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::sync::{Arc, Barrier};
    use std::thread;

    use bot_core::chat_config::ChatConfig;
    use postgres::Client;
    use postgres_native_tls::MakeTlsConnector;
    use serde_json::json;

    use super::{ChatConfigRepository, changed_fields};

    #[test]
    fn concurrent_config_updates_can_merge_only_the_fields_they_changed() {
        let previous = ChatConfig::default();
        let mut updated = previous.clone();
        updated.language = "en".to_owned();
        let patch = changed_fields(&previous, &updated);
        assert!(patch.is_ok());
        let Ok(patch) = patch else { return };
        assert_eq!(patch.len(), 1);
        assert_eq!(patch.get("language"), Some(&json!("en")));
    }

    #[test]
    fn concurrent_database_updates_preserve_unrelated_fields() {
        let Some(database_url) = test_database_url() else {
            return;
        };
        let chat_id = "-100900003";
        cleanup(&database_url, chat_id);
        let previous = ChatConfig::default();
        let mut language = previous.clone();
        language.language = "en".to_owned();
        let mut timezone = previous.clone();
        timezone.timezone_offset = -3;
        let barrier = Arc::new(Barrier::new(2));

        let first_url = database_url.clone();
        let first_barrier = barrier.clone();
        let first_previous = previous.clone();
        let first = thread::spawn(move || {
            first_barrier.wait();
            ChatConfigRepository::new(&first_url).set_changed(chat_id, &first_previous, &language)
        });
        let second_url = database_url.clone();
        let second = thread::spawn(move || {
            barrier.wait();
            ChatConfigRepository::new(&second_url).set_changed(chat_id, &previous, &timezone)
        });

        assert!(first.join().is_ok_and(|result| result.is_ok()));
        assert!(second.join().is_ok_and(|result| result.is_ok()));
        let loaded = ChatConfigRepository::new(&database_url).get(chat_id);
        assert!(loaded.is_ok());
        let Ok(Some(loaded)) = loaded else { return };
        assert_eq!(loaded.language, "en");
        assert_eq!(loaded.timezone_offset, -3);
        cleanup(&database_url, chat_id);
    }

    fn test_database_url() -> Option<String> {
        env::var("TEST_POSTGRES_URL").ok()
    }

    fn cleanup(database_url: &str, chat_id: &str) {
        let connector = native_tls::TlsConnector::builder().build();
        let Ok(connector) = connector else { return };
        let client = Client::connect(database_url, MakeTlsConnector::new(connector));
        let Ok(mut client) = client else { return };
        let _result = client.execute("DELETE FROM chat_configs WHERE chat_id = $1", &[&chat_id]);
    }

    #[test]
    fn reads_absent_and_native_rows_without_schema_changes() {
        let Some(database_url) = test_database_url() else {
            return;
        };
        let chat_id = "-100900001";
        cleanup(&database_url, chat_id);
        let repository = ChatConfigRepository::new(&database_url);
        assert!(repository.ensure_schema().is_ok());
        assert_eq!(repository.get(chat_id).ok(), Some(None));

        let connector = native_tls::TlsConnector::builder().build();
        assert!(connector.is_ok());
        let Ok(connector) = connector else { return };
        let client = Client::connect(&database_url, MakeTlsConnector::new(connector));
        assert!(client.is_ok());
        let Ok(mut client) = client else { return };
        let inserted = client.execute(
            "INSERT INTO chat_configs (chat_id, config) VALUES ($1, $2)",
            &[
                &chat_id,
                &json!({"language":"en", "creditless_user_hourly_limit":8}),
            ],
        );
        assert_eq!(inserted.ok(), Some(1));
        let loaded = repository.get(chat_id);
        assert!(loaded.is_ok());
        let Ok(Some(loaded)) = loaded else { return };
        assert_eq!(loaded.language, "en");
        assert_eq!(loaded.creditless_user_hourly_limit, 8);
        cleanup(&database_url, chat_id);
    }

    #[test]
    fn upsert_round_trip_is_idempotent_and_python_readable() {
        let Some(database_url) = test_database_url() else {
            return;
        };
        let chat_id = "-100900002";
        let repository = ChatConfigRepository::new(&database_url);
        assert!(repository.ensure_schema().is_ok());
        cleanup(&database_url, chat_id);
        let config = ChatConfig {
            language: "en".to_owned(),
            timezone_offset: 2,
            ai_random_replies: false,
            ..ChatConfig::default()
        };
        let connector = native_tls::TlsConnector::builder().build();
        assert!(connector.is_ok());
        let Ok(connector) = connector else { return };
        let client = Client::connect(&database_url, MakeTlsConnector::new(connector));
        assert!(client.is_ok());
        let Ok(mut client) = client else { return };
        assert_eq!(
            client
                .execute(
                    "INSERT INTO chat_configs (chat_id, config) VALUES ($1, $2)",
                    &[&chat_id, &json!({"future_setting":"preserved"})],
                )
                .ok(),
            Some(1)
        );
        assert_eq!(repository.set(chat_id, &config).ok(), Some(config.clone()));
        assert_eq!(repository.set(chat_id, &config).ok(), Some(config.clone()));
        assert_eq!(repository.get(chat_id).ok(), Some(Some(config.clone())));

        let row = client.query_one(
            "SELECT config FROM chat_configs WHERE chat_id = $1",
            &[&chat_id],
        );
        assert!(row.is_ok());
        let Ok(row) = row else { return };
        let stored: serde_json::Value = row.get(0);
        assert_eq!(stored.get("language"), Some(&json!("en")));
        assert_eq!(stored.get("timezone_offset"), Some(&json!(2)));
        assert_eq!(stored.get("ai_random_replies"), Some(&json!(false)));
        assert_eq!(stored.get("future_setting"), Some(&json!("preserved")));
        cleanup(&database_url, chat_id);
    }
}
