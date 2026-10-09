//! PostgreSQL repository for members group admins banned from the bot.

use std::sync::atomic::{AtomicBool, Ordering};

use bot_core::chat_bans::BannedUser;
use postgres::Client;
use thiserror::Error;

use crate::postgres_pool::{PooledPostgresClient, PostgresPool, PostgresPoolError};

const CHAT_BANS_SCHEMA_ADVISORY_LOCK_KEY: i64 = 48_610_007;

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS chat_bans (
    chat_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    banned_by BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (chat_id, user_id)
);
";

#[derive(Debug, Error)]
pub enum ChatBanRepositoryError {
    #[error("PostgreSQL chat ban operation failed: {0}{code}", code = crate::postgres_pool::sqlstate_suffix(.0))]
    Postgres(#[from] postgres::Error),
    #[error(transparent)]
    Pool(#[from] PostgresPoolError),
}

pub struct ChatBanRepository {
    pool: PostgresPool,
    /// Set once the table is known to exist, so the per-message ban check
    /// skips the schema transaction and its global advisory lock.
    schema_ready: AtomicBool,
}

impl ChatBanRepository {
    #[must_use]
    pub fn new(database_url: &str) -> Self {
        Self {
            pool: PostgresPool::shared(database_url),
            schema_ready: AtomicBool::new(false),
        }
    }

    fn connect(&self) -> Result<PooledPostgresClient, ChatBanRepositoryError> {
        let mut client = self.pool.get()?;
        if !self.schema_ready.load(Ordering::Acquire) {
            ensure_schema(&mut client)?;
            self.schema_ready.store(true, Ordering::Release);
        }
        Ok(client)
    }

    pub fn is_banned(&self, chat_id: i64, user_id: i64) -> Result<bool, ChatBanRepositoryError> {
        let mut client = self.connect()?;
        let row = client.query_opt(
            "SELECT 1 FROM chat_bans WHERE chat_id = $1 AND user_id = $2",
            &[&chat_id, &user_id],
        )?;
        Ok(row.is_some())
    }

    /// Returns whether the ban is new. Banning again keeps the first record.
    pub fn ban(
        &self,
        chat_id: i64,
        user_id: i64,
        display_name: &str,
        banned_by: i64,
    ) -> Result<bool, ChatBanRepositoryError> {
        let mut client = self.connect()?;
        let inserted = client.execute(
            "INSERT INTO chat_bans (chat_id, user_id, display_name, banned_by) \
             VALUES ($1, $2, $3, $4) ON CONFLICT (chat_id, user_id) DO NOTHING",
            &[&chat_id, &user_id, &display_name, &banned_by],
        )?;
        Ok(inserted == 1)
    }

    /// Returns whether a ban was lifted.
    pub fn unban(&self, chat_id: i64, user_id: i64) -> Result<bool, ChatBanRepositoryError> {
        let mut client = self.connect()?;
        let deleted = client.execute(
            "DELETE FROM chat_bans WHERE chat_id = $1 AND user_id = $2",
            &[&chat_id, &user_id],
        )?;
        Ok(deleted == 1)
    }

    /// Oldest ban first.
    pub fn list(&self, chat_id: i64) -> Result<Vec<BannedUser>, ChatBanRepositoryError> {
        let mut client = self.connect()?;
        let rows = client.query(
            "SELECT user_id, display_name FROM chat_bans WHERE chat_id = $1 \
             ORDER BY created_at, user_id",
            &[&chat_id],
        )?;
        Ok(rows
            .iter()
            .map(|row| BannedUser {
                user_id: row.get(0),
                name: row.get(1),
            })
            .collect())
    }
}

fn ensure_schema(client: &mut Client) -> Result<(), ChatBanRepositoryError> {
    let mut transaction = client.transaction()?;
    transaction.query_one(
        "SELECT pg_advisory_xact_lock($1)",
        &[&CHAT_BANS_SCHEMA_ADVISORY_LOCK_KEY],
    )?;
    transaction.batch_execute(SCHEMA_SQL)?;
    transaction.commit()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::env;

    use bot_core::chat_bans::BannedUser;
    use postgres::error::SqlState;

    use super::{ChatBanRepository, ChatBanRepositoryError};
    use crate::billing_schema::fault_injection;

    fn url() -> Option<String> {
        env::var("TEST_DATABASE_URL").ok()
    }

    fn cleanup(database_url: &str, chat_id: i64) {
        if let Ok(mut client) = fault_injection::connect(database_url) {
            let _result = client.execute("DELETE FROM chat_bans WHERE chat_id = $1", &[&chat_id]);
        }
    }

    #[test]
    fn bans_are_per_chat_idempotent_and_listed_oldest_first() {
        let Some(database_url) = url() else { return };
        let (chat_id, other_chat) = (-100_900_101, -100_900_102);
        let repository = ChatBanRepository::new(&database_url);
        // The first call creates the table, so leftovers from an interrupted
        // run can only be cleaned up after it.
        let _schema = repository.is_banned(chat_id, 2);
        cleanup(&database_url, chat_id);
        cleanup(&database_url, other_chat);
        assert_eq!(repository.list(chat_id).ok(), Some(Vec::new()));

        assert_eq!(repository.is_banned(chat_id, 2).ok(), Some(false));
        assert_eq!(repository.ban(chat_id, 2, "Ana", 1).ok(), Some(true));
        assert_eq!(
            repository.ban(chat_id, 2, "Ana renamed", 9).ok(),
            Some(false)
        );
        assert_eq!(repository.ban(chat_id, 3, "", 1).ok(), Some(true));
        assert_eq!(repository.is_banned(chat_id, 2).ok(), Some(true));
        assert_eq!(repository.is_banned(other_chat, 2).ok(), Some(false));
        assert_eq!(
            repository.list(chat_id).ok(),
            Some(vec![
                BannedUser {
                    user_id: 2,
                    name: "Ana".to_owned()
                },
                BannedUser {
                    user_id: 3,
                    name: String::new()
                },
            ])
        );
        assert_eq!(repository.list(other_chat).ok(), Some(Vec::new()));

        assert_eq!(repository.unban(other_chat, 2).ok(), Some(false));
        assert_eq!(repository.unban(chat_id, 2).ok(), Some(true));
        assert_eq!(repository.unban(chat_id, 2).ok(), Some(false));
        assert_eq!(repository.is_banned(chat_id, 2).ok(), Some(false));
        cleanup(&database_url, chat_id);
    }

    #[test]
    fn schema_setup_gives_up_while_another_setup_holds_the_lock() {
        let Some(url) = url() else { return };
        let holder = fault_injection::connect(&url);
        assert!(holder.is_ok());
        let Ok(mut holder) = holder else { return };
        let lock = "SELECT pg_advisory_lock(48610007)";
        assert!(holder.query_one(lock, &[]).is_ok());
        let separator = if url.contains('?') { '&' } else { '?' };
        let impatient_url = format!("{url}{separator}options=-clock_timeout%3D100");
        let impatient = ChatBanRepository::new(&impatient_url);
        let blocked = impatient.is_banned(-100_900_103, 2);
        let unlock = "SELECT pg_advisory_unlock(48610007)";
        assert!(holder.query_one(unlock, &[]).is_ok());
        assert!(
            matches!(&blocked, Err(ChatBanRepositoryError::Postgres(error))
            if error.code() == Some(&SqlState::LOCK_NOT_AVAILABLE))
        );
        assert!(blocked.is_err_and(|error| error.to_string().contains("chat ban")));
        // The failed setup is not remembered, so the next call retries it.
        assert_eq!(impatient.is_banned(-100_900_103, 2).ok(), Some(false));
    }

    #[test]
    fn unreachable_database_reports_a_pool_error() {
        let repository =
            ChatBanRepository::new("postgresql://synthetic:synthetic@127.0.0.1:1/none");
        assert!(matches!(
            repository.is_banned(1, 2),
            Err(ChatBanRepositoryError::Pool(_))
        ));
    }
}
