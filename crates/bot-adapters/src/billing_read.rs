//! Read-only PostgreSQL billing access used before billing writer cutover.

use native_tls::TlsConnector;
use postgres::{Client, error::SqlState};
use postgres_native_tls::MakeTlsConnector;
use serde::Serialize;
use serde_json::json;
use thiserror::Error;

const ONBOARDING_MAX_GRANTS_PER_HOUR: i64 = 4;
const ONBOARDING_MAX_GRANTS_PER_DAY: i64 = 16;
const ONBOARDING_GRANTS_ADVISORY_LOCK_KEY: i64 = 48_610_001;
const CREDIT_TRANSACTION_MAX_ATTEMPTS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BillingScope {
    User,
    Chat,
}

impl BillingScope {
    fn parse(value: &str) -> Result<Self, BillingError> {
        match value {
            "user" => Ok(Self::User),
            "chat" => Ok(Self::Chat),
            _ => Err(BillingError::InvalidScope(value.to_owned())),
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
pub enum BillingError {
    #[error("billing scope must be user or chat, got {0}")]
    InvalidScope(String),
    #[error("could not initialize PostgreSQL TLS: {0}")]
    Tls(#[from] native_tls::Error),
    #[error("PostgreSQL billing read failed: {0}")]
    Postgres(#[from] postgres::Error),
    #[error("PostgreSQL onboarding transaction retry limit was exhausted")]
    TransactionRetriesExhausted,
    #[error("billing balance exceeds the PostgreSQL integer range")]
    BalanceOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct OnboardingGrantResult {
    pub granted: bool,
    pub balance: i64,
}

pub struct BillingRepository {
    database_url: String,
}

impl BillingRepository {
    pub fn new(database_url: &str) -> Self {
        Self {
            database_url: database_url.to_owned(),
        }
    }

    pub fn get_balance(&self, scope_type: &str, scope_id: i64) -> Result<i64, BillingError> {
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
    ) -> Result<i64, BillingError> {
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

    pub fn grant_onboarding_if_needed(
        &self,
        user_id: i64,
        credits: i32,
    ) -> Result<OnboardingGrantResult, BillingError> {
        for attempt in 0..CREDIT_TRANSACTION_MAX_ATTEMPTS {
            match self.try_grant_onboarding(user_id, credits) {
                Ok(result) => return Ok(result),
                Err(error)
                    if attempt + 1 < CREDIT_TRANSACTION_MAX_ATTEMPTS
                        && is_retryable_transaction_error(&error) =>
                {
                    continue;
                }
                Err(error) => return Err(error),
            }
        }
        Err(BillingError::TransactionRetriesExhausted)
    }

    fn try_grant_onboarding(
        &self,
        user_id: i64,
        credits: i32,
    ) -> Result<OnboardingGrantResult, BillingError> {
        let mut client = self.connect()?;
        let mut transaction = client.transaction()?;
        transaction.query_one(
            "SELECT pg_advisory_xact_lock($1)",
            &[&ONBOARDING_GRANTS_ADVISORY_LOCK_KEY],
        )?;
        transaction.execute(
            "INSERT INTO credit_accounts (scope_type, scope_id, balance) VALUES ('user', $1, 0) \
             ON CONFLICT (scope_type, scope_id) DO NOTHING",
            &[&user_id],
        )?;
        let mut balance = transaction
            .query_one(
                "SELECT balance FROM credit_accounts \
                 WHERE scope_type = 'user' AND scope_id = $1 FOR UPDATE",
                &[&user_id],
            )?
            .get::<_, i32>(0);
        if transaction
            .query_opt(
                "SELECT 1 FROM onboarding_grants WHERE user_id = $1",
                &[&user_id],
            )?
            .is_some()
        {
            transaction.commit()?;
            return Ok(OnboardingGrantResult {
                granted: false,
                balance: balance.into(),
            });
        }

        let counts = transaction.query_one(
            "SELECT \
                COUNT(*) FILTER (WHERE granted_at >= NOW() - INTERVAL '1 hour'), \
                COUNT(*) FILTER (WHERE granted_at >= NOW() - INTERVAL '1 day') \
             FROM onboarding_grants",
            &[],
        )?;
        let hourly_count = counts.get::<_, i64>(0);
        let daily_count = counts.get::<_, i64>(1);
        if hourly_count >= ONBOARDING_MAX_GRANTS_PER_HOUR
            || daily_count >= ONBOARDING_MAX_GRANTS_PER_DAY
        {
            let metadata = json!({
                "credits": credits,
                "hourly_count": hourly_count,
                "daily_count": daily_count,
                "hourly_limit": ONBOARDING_MAX_GRANTS_PER_HOUR,
                "daily_limit": ONBOARDING_MAX_GRANTS_PER_DAY,
            });
            transaction.execute(
                "INSERT INTO credit_ledger \
                    (event_type, actor_user_id, user_id, amount, metadata) \
                 VALUES ('onboarding_denied_overflow', $1, $1, 0, $2)",
                &[&user_id, &metadata],
            )?;
            transaction.commit()?;
            return Ok(OnboardingGrantResult {
                granted: false,
                balance: balance.into(),
            });
        }

        let granted = transaction
            .query_opt(
                "INSERT INTO onboarding_grants (user_id, credits) VALUES ($1, $2) \
                 ON CONFLICT (user_id) DO NOTHING RETURNING user_id",
                &[&user_id, &credits],
            )?
            .is_some();
        if granted {
            balance = balance
                .checked_add(credits)
                .ok_or(BillingError::BalanceOverflow)?;
            transaction.execute(
                "UPDATE credit_accounts SET balance = $1, updated_at = NOW() \
                 WHERE scope_type = 'user' AND scope_id = $2",
                &[&balance, &user_id],
            )?;
            let metadata = json!({"credits": credits});
            transaction.execute(
                "INSERT INTO credit_ledger \
                    (event_type, actor_user_id, user_id, amount, metadata) \
                 VALUES ('onboarding_grant', $1, $1, $2, $3)",
                &[&user_id, &credits, &metadata],
            )?;
        }
        transaction.commit()?;
        Ok(OnboardingGrantResult {
            granted,
            balance: balance.into(),
        })
    }

    fn connect(&self) -> Result<Client, BillingError> {
        let connector = TlsConnector::builder().build()?;
        Ok(Client::connect(
            &self.database_url,
            MakeTlsConnector::new(connector),
        )?)
    }
}

fn is_retryable_transaction_error(error: &BillingError) -> bool {
    let BillingError::Postgres(error) = error else {
        return false;
    };
    error.code().is_some_and(|code| {
        *code == SqlState::T_R_SERIALIZATION_FAILURE || *code == SqlState::T_R_DEADLOCK_DETECTED
    })
}

#[cfg(test)]
mod tests {
    use native_tls::TlsConnector;
    use postgres::Client;
    use postgres_native_tls::MakeTlsConnector;

    use super::{BillingError, BillingRepository, BillingScope, OnboardingGrantResult};

    #[test]
    fn validates_the_persistent_scope_contract_before_connecting() {
        assert_eq!(BillingScope::parse("user").ok(), Some(BillingScope::User));
        assert_eq!(BillingScope::parse("chat").ok(), Some(BillingScope::Chat));
        let repository = BillingRepository::new("postgresql://invalid.invalid/db");
        assert!(matches!(
            repository.get_balance("group", 1),
            Err(BillingError::InvalidScope(scope)) if scope == "group"
        ));
        assert!(matches!(
            repository.get_or_create_balance("group", 1),
            Err(BillingError::InvalidScope(scope)) if scope == "group"
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
            CREATE TABLE IF NOT EXISTS onboarding_grants (\
                user_id BIGINT PRIMARY KEY, \
                credits INTEGER NOT NULL, \
                granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()\
            ); \
            CREATE TABLE IF NOT EXISTS credit_ledger (\
                id BIGSERIAL PRIMARY KEY, \
                event_type TEXT NOT NULL, \
                actor_user_id BIGINT, \
                user_id BIGINT, \
                chat_id BIGINT, \
                amount INTEGER NOT NULL, \
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb, \
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()\
            ); \
            INSERT INTO credit_accounts (scope_type, scope_id, balance) \
            VALUES ('user', 7000000000001, 1234) \
            ON CONFLICT (scope_type, scope_id) DO UPDATE SET balance = EXCLUDED.balance;",
        )?;
        drop(client);

        let repository = BillingRepository::new(&database_url);
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
        assert_eq!(
            repository.grant_onboarding_if_needed(7_000_000_000_003, 300)?,
            OnboardingGrantResult {
                granted: true,
                balance: 300,
            }
        );
        assert_eq!(
            repository.grant_onboarding_if_needed(7_000_000_000_003, 300)?,
            OnboardingGrantResult {
                granted: false,
                balance: 300,
            }
        );
        let first_database_url = database_url.clone();
        let second_database_url = database_url.clone();
        let first = std::thread::spawn(move || {
            BillingRepository::new(&first_database_url)
                .grant_onboarding_if_needed(7_000_000_000_004, 300)
        });
        let second = std::thread::spawn(move || {
            BillingRepository::new(&second_database_url)
                .grant_onboarding_if_needed(7_000_000_000_004, 300)
        });
        let concurrent_results = [
            first
                .join()
                .map_err(|_| std::io::Error::other("first onboarding thread panicked"))??,
            second
                .join()
                .map_err(|_| std::io::Error::other("second onboarding thread panicked"))??,
        ];
        assert_eq!(
            concurrent_results
                .iter()
                .filter(|result| result.granted)
                .count(),
            1
        );
        assert!(
            concurrent_results
                .iter()
                .all(|result| result.balance == 300)
        );
        client.batch_execute(
            "INSERT INTO onboarding_grants (user_id, credits) VALUES \
                (7000000000004, 1), (7000000000005, 1), (7000000000006, 1) \
             ON CONFLICT (user_id) DO NOTHING;",
        )?;
        assert_eq!(
            repository.grant_onboarding_if_needed(7_000_000_000_007, 300)?,
            OnboardingGrantResult {
                granted: false,
                balance: 0,
            }
        );
        let event_counts = client.query_one(
            "SELECT \
                COUNT(*) FILTER (WHERE event_type = 'onboarding_grant'), \
                COUNT(*) FILTER (WHERE event_type = 'onboarding_denied_overflow') \
             FROM credit_ledger WHERE user_id >= $1 AND user_id <= $2",
            &[&7_000_000_000_001_i64, &7_000_000_000_007_i64],
        )?;
        assert_eq!(event_counts.get::<_, i64>(0), 2);
        assert_eq!(event_counts.get::<_, i64>(1), 1);

        let synthetic_ids = [
            7_000_000_000_001_i64,
            7_000_000_000_002_i64,
            7_000_000_000_003_i64,
            7_000_000_000_004_i64,
            7_000_000_000_005_i64,
            7_000_000_000_006_i64,
            7_000_000_000_007_i64,
        ];
        client.execute(
            "DELETE FROM credit_ledger WHERE user_id = ANY($1)",
            &[&&synthetic_ids[..]],
        )?;
        client.execute(
            "DELETE FROM onboarding_grants WHERE user_id = ANY($1)",
            &[&&synthetic_ids[..]],
        )?;
        client.execute(
            "DELETE FROM credit_accounts WHERE scope_id = ANY($1)",
            &[&&synthetic_ids[..]],
        )?;
        Ok(())
    }
}
