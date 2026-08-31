//! PostgreSQL chat configuration repository compatible with the Python schema.

use bot_core::chat_config::{ChatConfig, ChatConfigError};
use native_tls::TlsConnector;
use postgres::Client;
use postgres_native_tls::MakeTlsConnector;
use serde_json::Value;
use thiserror::Error;

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
}

pub struct ChatConfigRepository {
    database_url: String,
}

impl ChatConfigRepository {
    #[must_use]
    pub fn new(database_url: &str) -> Self {
        Self {
            database_url: database_url.to_owned(),
        }
    }

    pub fn ensure_schema(&self) -> Result<(), ChatConfigRepositoryError> {
        self.connect()?.batch_execute(SCHEMA_SQL)?;
        Ok(())
    }

    pub fn get(&self, chat_id: &str) -> Result<Option<ChatConfig>, ChatConfigRepositoryError> {
        let mut client = self.connect()?;
        client.batch_execute(SCHEMA_SQL)?;
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
        let mut client = self.connect()?;
        client.batch_execute(SCHEMA_SQL)?;
        let value = serde_json::to_value(config)?;
        client.execute(
            "INSERT INTO chat_configs (chat_id, config) VALUES ($1, $2) \
             ON CONFLICT (chat_id) DO UPDATE SET \
             config = chat_configs.config || EXCLUDED.config, updated_at = NOW()",
            &[&chat_id, &value],
        )?;
        Ok(config.clone())
    }

    fn connect(&self) -> Result<Client, ChatConfigRepositoryError> {
        let connector = TlsConnector::builder().build()?;
        Ok(Client::connect(
            &self.database_url,
            MakeTlsConnector::new(connector),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use std::env;

    use bot_core::chat_config::ChatConfig;
    use postgres::Client;
    use postgres_native_tls::MakeTlsConnector;
    use serde_json::json;

    use super::ChatConfigRepository;

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
    fn reads_absent_legacy_and_native_rows_without_schema_changes() {
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
                &json!({"language":"en", "creditless_user_daily_limit":8}),
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
