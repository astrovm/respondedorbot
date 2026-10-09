//! PostgreSQL repository for members group admins gave their own hourly
//! limit of AI messages paid by the group.

use std::sync::atomic::{AtomicBool, Ordering};

use bot_core::chat_limits::LimitedUser;
use postgres::Client;
use thiserror::Error;

use crate::postgres_pool::{PooledPostgresClient, PostgresPool, PostgresPoolError};

const CHAT_LIMITS_SCHEMA_ADVISORY_LOCK_KEY: i64 = 48_610_008;

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS chat_user_limits (
    chat_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    hourly_limit BIGINT NOT NULL CHECK (hourly_limit >= 0),
    display_name TEXT NOT NULL DEFAULT '',
    set_by BIGINT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (chat_id, user_id)
);
";

#[derive(Debug, Error)]
pub enum ChatLimitRepositoryError {
    #[error("PostgreSQL chat limit operation failed: {0}{code}", code = crate::postgres_pool::sqlstate_suffix(.0))]
    Postgres(#[from] postgres::Error),
    #[error(transparent)]
    Pool(#[from] PostgresPoolError),
}

pub struct ChatLimitRepository {
    pool: PostgresPool,
    /// Set once the table is known to exist, so the per-reply limit lookup
    /// skips the schema transaction and its global advisory lock.
    schema_ready: AtomicBool,
}

impl ChatLimitRepository {
    #[must_use]
    pub fn new(database_url: &str) -> Self {
        Self {
            pool: PostgresPool::shared(database_url),
            schema_ready: AtomicBool::new(false),
        }
    }

    fn connect(&self) -> Result<PooledPostgresClient, ChatLimitRepositoryError> {
        let mut client = self.pool.get()?;
        if !self.schema_ready.load(Ordering::Acquire) {
            ensure_schema(&mut client)?;
            self.schema_ready.store(true, Ordering::Release);
        }
        Ok(client)
    }

    pub fn hourly_limit(
        &self,
        chat_id: i64,
        user_id: i64,
    ) -> Result<Option<i64>, ChatLimitRepositoryError> {
        let mut client = self.connect()?;
        let row = client.query_opt(
            "SELECT hourly_limit FROM chat_user_limits WHERE chat_id = $1 AND user_id = $2",
            &[&chat_id, &user_id],
        )?;
        Ok(row.map(|row| row.get(0)))
    }

    /// Setting again replaces the previous limit and name.
    pub fn set(
        &self,
        chat_id: i64,
        user_id: i64,
        display_name: &str,
        hourly_limit: i64,
        set_by: i64,
    ) -> Result<(), ChatLimitRepositoryError> {
        let mut client = self.connect()?;
        client.execute(
            "INSERT INTO chat_user_limits \
                (chat_id, user_id, hourly_limit, display_name, set_by) \
             VALUES ($1, $2, $3, $4, $5) ON CONFLICT (chat_id, user_id) DO UPDATE \
             SET hourly_limit = EXCLUDED.hourly_limit, display_name = EXCLUDED.display_name, \
                 set_by = EXCLUDED.set_by, updated_at = NOW()",
            &[&chat_id, &user_id, &hourly_limit, &display_name, &set_by],
        )?;
        Ok(())
    }

    /// Returns whether a limit was removed.
    pub fn clear(&self, chat_id: i64, user_id: i64) -> Result<bool, ChatLimitRepositoryError> {
        let mut client = self.connect()?;
        let deleted = client.execute(
            "DELETE FROM chat_user_limits WHERE chat_id = $1 AND user_id = $2",
            &[&chat_id, &user_id],
        )?;
        Ok(deleted == 1)
    }

    /// Least recently set first.
    pub fn list(&self, chat_id: i64) -> Result<Vec<LimitedUser>, ChatLimitRepositoryError> {
        let mut client = self.connect()?;
        let rows = client.query(
            "SELECT user_id, display_name, hourly_limit FROM chat_user_limits \
             WHERE chat_id = $1 ORDER BY updated_at, user_id",
            &[&chat_id],
        )?;
        Ok(rows
            .iter()
            .map(|row| LimitedUser {
                user_id: row.get(0),
                name: row.get(1),
                hourly_limit: row.get(2),
            })
            .collect())
    }
}

fn ensure_schema(client: &mut Client) -> Result<(), ChatLimitRepositoryError> {
    let mut transaction = client.transaction()?;
    transaction.query_one(
        "SELECT pg_advisory_xact_lock($1)",
        &[&CHAT_LIMITS_SCHEMA_ADVISORY_LOCK_KEY],
    )?;
    transaction.batch_execute(SCHEMA_SQL)?;
    transaction.commit()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::env;

    use bot_core::chat_limits::LimitedUser;
    use postgres::error::SqlState;

    use super::{ChatLimitRepository, ChatLimitRepositoryError};
    use crate::billing_schema::fault_injection;

    fn url() -> Option<String> {
        env::var("TEST_DATABASE_URL").ok()
    }

    fn cleanup(database_url: &str, chat_id: i64) {
        if let Ok(mut client) = fault_injection::connect(database_url) {
            let _result = client.execute(
                "DELETE FROM chat_user_limits WHERE chat_id = $1",
                &[&chat_id],
            );
        }
    }

    #[test]
    fn limits_are_per_chat_replaced_and_cleared() {
        let Some(database_url) = url() else { return };
        let (chat_id, other_chat) = (-100_900_111, -100_900_112);
        let repository = ChatLimitRepository::new(&database_url);
        // The first lookup creates the table the cleanup deletes from.
        assert!(repository.hourly_limit(chat_id, 2).is_ok());
        cleanup(&database_url, chat_id);
        cleanup(&database_url, other_chat);

        assert_eq!(repository.list(chat_id).ok(), Some(Vec::new()));
        assert_eq!(repository.set(chat_id, 2, "Ana", 3, 1).ok(), Some(()));
        assert_eq!(repository.hourly_limit(chat_id, 2).ok(), Some(Some(3)));
        assert_eq!(repository.set(chat_id, 3, "", 5, 1).ok(), Some(()));
        assert_eq!(repository.set(chat_id, 2, "Ana B", 0, 9).ok(), Some(()));
        assert_eq!(repository.hourly_limit(chat_id, 2).ok(), Some(Some(0)));
        assert_eq!(repository.hourly_limit(other_chat, 2).ok(), Some(None));
        assert_eq!(repository.hourly_limit(chat_id, 4).ok(), Some(None));
        // Setting again moves the member to the end of the list.
        assert_eq!(
            repository.list(chat_id).ok(),
            Some(vec![
                LimitedUser {
                    user_id: 3,
                    name: String::new(),
                    hourly_limit: 5
                },
                LimitedUser {
                    user_id: 2,
                    name: "Ana B".to_owned(),
                    hourly_limit: 0
                },
            ])
        );
        assert_eq!(repository.list(other_chat).ok(), Some(Vec::new()));
        assert_eq!(repository.clear(chat_id, 3).ok(), Some(true));

        assert_eq!(repository.clear(other_chat, 2).ok(), Some(false));
        assert_eq!(repository.clear(chat_id, 2).ok(), Some(true));
        assert_eq!(repository.clear(chat_id, 2).ok(), Some(false));
        assert_eq!(repository.hourly_limit(chat_id, 2).ok(), Some(None));
        cleanup(&database_url, chat_id);
    }

    #[test]
    fn negative_limits_are_rejected_by_the_table() {
        let Some(database_url) = url() else { return };
        let repository = ChatLimitRepository::new(&database_url);
        let rejected = repository.set(-100_900_113, 2, "Ana", -1, 1);
        assert!(
            matches!(&rejected, Err(ChatLimitRepositoryError::Postgres(error))
            if error.code() == Some(&SqlState::CHECK_VIOLATION))
        );
        assert_eq!(repository.hourly_limit(-100_900_113, 2).ok(), Some(None));
    }

    #[test]
    fn schema_setup_gives_up_while_another_setup_holds_the_lock() {
        let Some(url) = url() else { return };
        let holder = fault_injection::connect(&url);
        assert!(holder.is_ok());
        let Ok(mut holder) = holder else { return };
        let lock = "SELECT pg_advisory_lock(48610008)";
        assert!(holder.query_one(lock, &[]).is_ok());
        let separator = if url.contains('?') { '&' } else { '?' };
        let impatient_url = format!("{url}{separator}options=-clock_timeout%3D100");
        let impatient = ChatLimitRepository::new(&impatient_url);
        let blocked = impatient.hourly_limit(-100_900_114, 2);
        let unlock = "SELECT pg_advisory_unlock(48610008)";
        assert!(holder.query_one(unlock, &[]).is_ok());
        assert!(
            matches!(&blocked, Err(ChatLimitRepositoryError::Postgres(error))
            if error.code() == Some(&SqlState::LOCK_NOT_AVAILABLE))
        );
        assert!(blocked.is_err_and(|error| error.to_string().contains("chat limit")));
        // The failed setup is not remembered, so the next call retries it.
        assert_eq!(impatient.hourly_limit(-100_900_114, 2).ok(), Some(None));
    }

    #[test]
    fn unreachable_database_reports_a_pool_error() {
        let repository =
            ChatLimitRepository::new("postgresql://synthetic:synthetic@127.0.0.1:1/none");
        assert!(matches!(
            repository.hourly_limit(1, 2),
            Err(ChatLimitRepositoryError::Pool(_))
        ));
    }
}
