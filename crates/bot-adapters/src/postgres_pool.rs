//! Small shared synchronous PostgreSQL connection pool.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use postgres::{Client, Config};
use thiserror::Error;

use crate::idle_pool::{IdleList, MAX_IDLE_AGE};
use crate::postgres_connection::postgres_tls_connector;

const MAX_IDLE_CONNECTIONS: usize = 16;

#[derive(Debug, Error)]
pub enum PostgresPoolError {
    #[error("could not initialize PostgreSQL TLS: {0}")]
    Tls(#[from] native_tls::Error),
    #[error("could not open PostgreSQL connection: {0}")]
    Postgres(#[from] postgres::Error),
}

struct PostgresPoolInner {
    database_url: String,
    idle: Mutex<IdleList<Client>>,
}

#[derive(Clone)]
pub struct PostgresPool {
    inner: Arc<PostgresPoolInner>,
}

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

pub struct PooledPostgresClient {
    client: Option<Client>,
    pool: Arc<PostgresPoolInner>,
}

impl PostgresPool {
    #[must_use]
    pub fn shared(database_url: &str) -> Self {
        static POOLS: OnceLock<Mutex<HashMap<String, Weak<PostgresPoolInner>>>> = OnceLock::new();
        let pools = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
        if let Ok(mut pools) = pools.lock() {
            if let Some(inner) = pools.get(database_url).and_then(Weak::upgrade) {
                return Self { inner };
            }
            let inner = Arc::new(PostgresPoolInner {
                database_url: database_url.to_owned(),
                idle: Mutex::new(IdleList::new()),
            });
            pools.insert(database_url.to_owned(), Arc::downgrade(&inner));
            return Self { inner };
        }
        Self {
            inner: Arc::new(PostgresPoolInner {
                database_url: database_url.to_owned(),
                idle: Mutex::new(IdleList::new()),
            }),
        }
    }

    pub fn get(&self) -> Result<PooledPostgresClient, PostgresPoolError> {
        let client = self
            .inner
            .idle
            .lock()
            .ok()
            .and_then(|mut idle| idle.pop_fresh(Instant::now(), MAX_IDLE_AGE))
            .map_or_else(
                || -> Result<Client, PostgresPoolError> {
                    let mut config = self.inner.database_url.parse::<Config>()?;
                    // Fail fast when the database is unreachable instead of
                    // blocking a worker for the operating system's TCP timeout.
                    if config.get_connect_timeout().is_none() {
                        config.connect_timeout(CONNECT_TIMEOUT);
                    }
                    Ok(config.connect(postgres_tls_connector(&self.inner.database_url)?)?)
                },
                Ok,
            )?;
        Ok(PooledPostgresClient {
            client: Some(client),
            pool: self.inner.clone(),
        })
    }
}

impl Deref for PooledPostgresClient {
    type Target = Client;

    fn deref(&self) -> &Self::Target {
        self.client
            .as_ref()
            .unwrap_or_else(|| unreachable!("pooled PostgreSQL client is present until drop"))
    }
}

impl DerefMut for PooledPostgresClient {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.client
            .as_mut()
            .unwrap_or_else(|| unreachable!("pooled PostgreSQL client is present until drop"))
    }
}

impl Drop for PooledPostgresClient {
    fn drop(&mut self) {
        let Some(client) = self.client.take() else {
            return;
        };
        if !client.is_closed()
            && let Ok(mut idle) = self.pool.idle.lock()
            && idle.len() < MAX_IDLE_CONNECTIONS
        {
            idle.push(Instant::now(), client);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Instant;

    use super::{MAX_IDLE_CONNECTIONS, PostgresPool, PostgresPoolError};

    /// A pool of its own: pools are shared per URL, so the application name
    /// keeps these tests away from other tests' idle connections.
    fn test_pool(name: &str) -> Option<PostgresPool> {
        let database_url = std::env::var("TEST_DATABASE_URL").ok()?;
        let separator = if database_url.contains('?') { '&' } else { '?' };
        Some(PostgresPool::shared(&format!(
            "{database_url}{separator}application_name=pool_test_{name}"
        )))
    }

    fn backend_pid(pool: &PostgresPool) -> Result<i32, Box<dyn std::error::Error>> {
        Ok(pool
            .get()?
            .query_one("SELECT pg_backend_pid()", &[])?
            .get(0))
    }

    #[test]
    fn shared_pools_are_reused_per_url_until_every_handle_drops() {
        let first = PostgresPool::shared("postgresql://synthetic.invalid/shared_a");
        let same = PostgresPool::shared("postgresql://synthetic.invalid/shared_a");
        let other = PostgresPool::shared("postgresql://synthetic.invalid/shared_b");
        assert!(Arc::ptr_eq(&first.inner, &same.inner));
        assert!(!Arc::ptr_eq(&first.inner, &other.inner));

        let weak = Arc::downgrade(&first.inner);
        drop((first, same));
        assert!(weak.upgrade().is_none());
        let fresh = PostgresPool::shared("postgresql://synthetic.invalid/shared_a");
        assert_eq!(
            fresh.inner.database_url,
            "postgresql://synthetic.invalid/shared_a"
        );
    }

    #[test]
    fn invalid_and_unreachable_databases_fail_without_hanging() {
        let invalid = PostgresPool::shared("not a database url");
        assert!(matches!(invalid.get(), Err(PostgresPoolError::Postgres(_))));

        let started = Instant::now();
        let refused = PostgresPool::shared(
            "postgresql://synthetic@127.0.0.1:1/synthetic?sslmode=disable&connect_timeout=2",
        );
        assert!(matches!(refused.get(), Err(PostgresPoolError::Postgres(_))));
        assert!(started.elapsed().as_secs() < 5);
    }

    #[test]
    fn returned_connections_are_reused() -> Result<(), Box<dyn std::error::Error>> {
        let Some(pool) = test_pool("reuse") else {
            return Ok(());
        };
        let first = backend_pid(&pool)?;
        assert_eq!(backend_pid(&pool)?, first);

        let held = pool.get()?;
        let concurrent = backend_pid(&pool)?;
        assert_ne!(concurrent, first);
        drop(held);
        Ok(())
    }

    #[test]
    fn closed_connections_are_discarded_instead_of_returned()
    -> Result<(), Box<dyn std::error::Error>> {
        let Some(pool) = test_pool("closed") else {
            return Ok(());
        };
        let terminated = backend_pid(&pool)?;
        let mut client = pool.get()?;
        let killer = PostgresPool::shared(&format!("{}_killer", pool.inner.database_url));
        killer
            .get()?
            .execute("SELECT pg_terminate_backend($1)", &[&terminated])?;
        assert!(client.query_one("SELECT 1", &[]).is_err());
        assert!(client.is_closed());
        drop(client);

        assert_ne!(backend_pid(&pool)?, terminated);
        Ok(())
    }

    #[test]
    fn idle_connections_are_capped() -> Result<(), Box<dyn std::error::Error>> {
        let Some(pool) = test_pool("cap") else {
            return Ok(());
        };
        let clients = (0..=MAX_IDLE_CONNECTIONS)
            .map(|_| pool.get())
            .collect::<Result<Vec<_>, _>>()?;
        drop(clients);
        let idle = pool.inner.idle.lock().map_or(0, |idle| idle.len());
        assert_eq!(idle, MAX_IDLE_CONNECTIONS);
        Ok(())
    }
}
