//! Read-only PostgreSQL billing access used before billing writer cutover.

use native_tls::TlsConnector;
use postgres::Client;
use postgres_native_tls::MakeTlsConnector;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BillingScope {
    User,
    Chat,
}

impl BillingScope {
    fn parse(value: &str) -> Result<Self, BillingReadError> {
        match value {
            "user" => Ok(Self::User),
            "chat" => Ok(Self::Chat),
            _ => Err(BillingReadError::InvalidScope(value.to_owned())),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Chat => "chat",
        }
    }
}

#[derive(Debug, Error)]
pub enum BillingReadError {
    #[error("billing scope must be user or chat, got {0}")]
    InvalidScope(String),
    #[error("could not initialize PostgreSQL TLS: {0}")]
    Tls(#[from] native_tls::Error),
    #[error("PostgreSQL billing read failed: {0}")]
    Postgres(#[from] postgres::Error),
}

pub struct BillingReadRepository {
    database_url: String,
}

impl BillingReadRepository {
    pub fn new(database_url: &str) -> Self {
        Self {
            database_url: database_url.to_owned(),
        }
    }

    pub fn get_balance(&self, scope_type: &str, scope_id: i64) -> Result<i64, BillingReadError> {
        let scope = BillingScope::parse(scope_type)?;
        let mut client = self.connect()?;
        let row = client.query_opt(
            "SELECT balance FROM credit_accounts WHERE scope_type = $1 AND scope_id = $2",
            &[&scope.as_str(), &scope_id],
        )?;
        Ok(row.map_or(0, |value| value.get::<_, i32>(0).into()))
    }

    pub fn get_or_create_balance(
        &self,
        scope_type: &str,
        scope_id: i64,
    ) -> Result<i64, BillingReadError> {
        let scope = BillingScope::parse(scope_type)?;
        let mut client = self.connect()?;
        let mut transaction = client.transaction()?;
        transaction.execute(
            "INSERT INTO credit_accounts (scope_type, scope_id, balance) VALUES ($1, $2, 0) \
             ON CONFLICT (scope_type, scope_id) DO NOTHING",
            &[&scope.as_str(), &scope_id],
        )?;
        let row = transaction.query_one(
            "SELECT balance FROM credit_accounts WHERE scope_type = $1 AND scope_id = $2",
            &[&scope.as_str(), &scope_id],
        )?;
        let balance = row.get::<_, i32>(0).into();
        transaction.commit()?;
        Ok(balance)
    }

    fn connect(&self) -> Result<Client, BillingReadError> {
        let connector = TlsConnector::builder().build()?;
        Ok(Client::connect(
            &self.database_url,
            MakeTlsConnector::new(connector),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use native_tls::TlsConnector;
    use postgres::Client;
    use postgres_native_tls::MakeTlsConnector;

    use super::{BillingReadError, BillingReadRepository, BillingScope};

    #[test]
    fn validates_the_persistent_scope_contract_before_connecting() {
        assert_eq!(BillingScope::parse("user").ok(), Some(BillingScope::User));
        assert_eq!(BillingScope::parse("chat").ok(), Some(BillingScope::Chat));
        let repository = BillingReadRepository::new("postgresql://invalid.invalid/db");
        assert!(matches!(
            repository.get_balance("group", 1),
            Err(BillingReadError::InvalidScope(scope)) if scope == "group"
        ));
        assert!(matches!(
            repository.get_or_create_balance("group", 1),
            Err(BillingReadError::InvalidScope(scope)) if scope == "group"
        ));
    }

    #[test]
    fn reads_existing_and_missing_balances_when_test_postgres_is_available()
    -> Result<(), Box<dyn std::error::Error>> {
        let Ok(database_url) = std::env::var("TEST_POSTGRES_URL") else {
            return Ok(());
        };
        let connector = TlsConnector::builder().build()?;
        let mut client = Client::connect(&database_url, MakeTlsConnector::new(connector))?;
        client.batch_execute(
            "CREATE TABLE IF NOT EXISTS credit_accounts (\
                scope_type TEXT NOT NULL, \
                scope_id BIGINT NOT NULL, \
                balance INTEGER NOT NULL, \
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), \
                PRIMARY KEY (scope_type, scope_id)\
            ); \
            INSERT INTO credit_accounts VALUES ('user', 7000000000001, 1234) \
            ON CONFLICT (scope_type, scope_id) DO UPDATE SET balance = EXCLUDED.balance;",
        )?;
        drop(client);

        let repository = BillingReadRepository::new(&database_url);
        assert_eq!(repository.get_balance("user", 7_000_000_000_001)?, 1234);
        assert_eq!(repository.get_balance("chat", 7_000_000_000_002)?, 0);
        assert_eq!(
            repository.get_or_create_balance("chat", 7_000_000_000_002)?,
            0
        );

        let connector = TlsConnector::builder().build()?;
        let mut client = Client::connect(&database_url, MakeTlsConnector::new(connector))?;
        assert_eq!(
            client
                .query_one(
                    "SELECT COUNT(*) FROM credit_accounts WHERE scope_type = 'chat' AND scope_id = $1",
                    &[&7_000_000_000_002_i64],
                )?
                .get::<_, i64>(0),
            1
        );
        let synthetic_ids = [7_000_000_000_001_i64, 7_000_000_000_002_i64];
        client.execute(
            "DELETE FROM credit_accounts WHERE scope_id = ANY($1)",
            &[&&synthetic_ids[..]],
        )?;
        Ok(())
    }
}
