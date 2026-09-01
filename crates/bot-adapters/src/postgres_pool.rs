//! Small shared synchronous PostgreSQL connection pool.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use postgres::Client;
use thiserror::Error;

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
    idle: Mutex<Vec<Client>>,
}

#[derive(Clone)]
pub struct PostgresPool {
    inner: Arc<PostgresPoolInner>,
}

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
                idle: Mutex::new(Vec::new()),
            });
            pools.insert(database_url.to_owned(), Arc::downgrade(&inner));
            return Self { inner };
        }
        Self {
            inner: Arc::new(PostgresPoolInner {
                database_url: database_url.to_owned(),
                idle: Mutex::new(Vec::new()),
            }),
        }
    }

    pub fn get(&self) -> Result<PooledPostgresClient, PostgresPoolError> {
        let client = self
            .inner
            .idle
            .lock()
            .ok()
            .and_then(|mut idle| idle.pop())
            .map_or_else(
                || -> Result<Client, PostgresPoolError> {
                    Ok(Client::connect(
                        &self.inner.database_url,
                        postgres_tls_connector(&self.inner.database_url)?,
                    )?)
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
            idle.push(client);
        }
    }
}
