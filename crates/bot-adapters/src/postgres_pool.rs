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
