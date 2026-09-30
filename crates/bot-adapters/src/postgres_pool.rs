//! Small shared synchronous PostgreSQL connection pool.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};
use std::time::{Duration, Instant};

use postgres::{Client, Config};
use thiserror::Error;

use crate::idle_pool::{IdleList, MAX_IDLE_AGE};
use crate::postgres_connection::postgres_tls_connector;

const MAX_IDLE_CONNECTIONS: usize = 16;
const CONNECTION_RETRY_DELAYS: [Duration; 2] = [Duration::from_secs(1), Duration::from_secs(2)];

#[derive(Debug, Error)]
pub enum PostgresPoolError {
    #[error("could not initialize PostgreSQL TLS: {0}")]
    Tls(#[from] native_tls::Error),
    #[error("could not open PostgreSQL connection: {0}{code}", code = sqlstate_suffix(.0))]
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
        // The map is only read and extended while locked, so a poisoned lock
        // still guards a consistent map.
        let mut pools = pools.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(inner) = pools.get(database_url).and_then(Weak::upgrade) {
            return Self { inner };
        }
        let inner = Arc::new(PostgresPoolInner {
            database_url: database_url.to_owned(),
            idle: Mutex::new(IdleList::new()),
        });
        pools.insert(database_url.to_owned(), Arc::downgrade(&inner));
        Self { inner }
    }

    pub fn get(&self) -> Result<PooledPostgresClient, PostgresPoolError> {
        let idle_client = self
            .inner
            .idle
            .lock()
            .ok()
            .and_then(|mut idle| idle.pop_fresh(Instant::now(), MAX_IDLE_AGE))
            .and_then(|mut client| client.is_valid(CONNECT_TIMEOUT).is_ok().then_some(client));
        let client = match idle_client {
            // The blocking client may not notice a pooler disconnect until it
            // drives I/O again. Validate before handing it to a business query;
            // failed writes and billing transactions must never be replayed here.
            Some(client) => client,
            _ => {
                let mut config = self.inner.database_url.parse::<Config>()?;
                if config.get_connect_timeout().is_none() {
                    config.connect_timeout(CONNECT_TIMEOUT);
                }
                let tls = postgres_tls_connector(&self.inner.database_url)?;
                connect_with_retry(|| config.connect(tls.clone()), std::thread::sleep)?
            }
        };
        Ok(PooledPostgresClient {
            client: Some(client),
            pool: self.inner.clone(),
        })
    }
}

pub(crate) fn sqlstate_suffix(error: &postgres::Error) -> String {
    error
        .code()
        .map_or_else(String::new, |code| format!(" (SQLSTATE {})", code.code()))
}

fn connect_with_retry<T>(
    mut connect: impl FnMut() -> Result<T, postgres::Error>,
    mut wait: impl FnMut(Duration),
) -> Result<T, postgres::Error> {
    for delay in CONNECTION_RETRY_DELAYS {
        match connect() {
            Ok(client) => return Ok(client),
            Err(error) if retryable_connection_error(&error) => wait(delay),
            Err(error) => return Err(error),
        }
    }
    connect()
}

fn retryable_connection_error(error: &postgres::Error) -> bool {
    error.code().is_none_or(|code| {
        code.code().starts_with("08")
            || matches!(code.code(), "53300" | "57P01" | "57P02" | "57P03")
    })
}

impl Deref for PooledPostgresClient {
    type Target = Client;

    #[allow(
        clippy::expect_used,
        reason = "the client is only taken by drop, after which it cannot be dereferenced"
    )]
    fn deref(&self) -> &Self::Target {
        self.client
            .as_ref()
            .expect("pooled PostgreSQL client is present until drop")
    }
}

impl DerefMut for PooledPostgresClient {
    #[allow(
        clippy::expect_used,
        reason = "the client is only taken by drop, after which it cannot be dereferenced"
    )]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.client
            .as_mut()
            .expect("pooled PostgreSQL client is present until drop")
    }
}

impl Drop for PooledPostgresClient {
    fn drop(&mut self) {
        if let Some(client) = self.client.take()
            && !client.is_closed()
            && let Ok(mut idle) = self.pool.idle.lock()
            && idle.len() < MAX_IDLE_CONNECTIONS
        {
            idle.push(Instant::now(), client);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

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
        test_pool("reuse").map_or(Ok(()), |pool| {
            let first = backend_pid(&pool)?;
            assert_eq!(backend_pid(&pool)?, first);

            let held = pool.get()?;
            let concurrent = backend_pid(&pool)?;
            assert_ne!(concurrent, first);
            drop(held);
            Ok(())
        })
    }

    #[test]
    fn checkout_replaces_connections_terminated_while_idle()
    -> Result<(), Box<dyn std::error::Error>> {
        test_pool("idle_disconnect").map_or(Ok(()), |pool| {
            let terminated = backend_pid(&pool)?;
            let killer = PostgresPool::shared(&format!("{}_killer", pool.inner.database_url));
            let row = killer
                .get()?
                .query_one("SELECT pg_terminate_backend($1, 5000)", &[&terminated])?;
            assert!(row.get::<_, bool>(0));
            // The first business query succeeds on a new connection, without
            // making the caller retry its operation.
            assert_ne!(backend_pid(&pool)?, terminated);
            Ok(())
        })
    }

    fn connection_error(code: &str) -> Result<postgres::Error, Box<dyn std::error::Error>> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let mut body = format!("SFATAL\0C{code}\0Msynthetic connection refusal\0").into_bytes();
        body.push(0);
        let server = thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            let mut size = [0; 4];
            stream.read_exact(&mut size)?;
            let mut startup = vec![0; u32::from_be_bytes(size) as usize - 4];
            stream.read_exact(&mut startup)?;
            stream.write_all(b"E")?;
            stream.write_all(&((body.len() + 4) as u32).to_be_bytes())?;
            stream.write_all(&body)
        });
        let result = postgres::Client::connect(
            &format!("host=127.0.0.1 port={port} user=synthetic sslmode=disable"),
            postgres::NoTls,
        );
        server
            .join()
            .map_err(|_| "synthetic PostgreSQL server panicked")??;
        result
            .err()
            .ok_or_else(|| "expected a connection refusal".into())
    }

    #[test]
    fn connection_retries_recover_transient_failures_and_stop_on_bad_credentials()
    -> Result<(), Box<dyn std::error::Error>> {
        for code in ["08006", "53300", "57P01", "57P02", "57P03"] {
            let mut outcomes = VecDeque::from([
                Err(connection_error(code)?),
                Err(connection_error(code)?),
                Ok("connected"),
            ]);
            let mut waits = Vec::new();
            let recovered = super::connect_with_retry(
                || {
                    outcomes
                        .pop_front()
                        .unwrap_or(Ok("unexpected extra attempt"))
                },
                |delay| waits.push(delay),
            );
            assert_eq!(recovered?, "connected");
            assert_eq!(waits, [Duration::from_secs(1), Duration::from_secs(2)]);
            assert!(outcomes.is_empty());
        }

        let mut attempts = 0;
        let mut outcomes = VecDeque::from([Err(connection_error("28P01")?), Ok(())]);
        let mut waits = Vec::new();
        let refused = super::connect_with_retry::<()>(
            || {
                attempts += 1;
                outcomes.pop_front().unwrap_or(Ok(()))
            },
            |delay| waits.push(delay),
        );
        assert_eq!(attempts, 1);
        assert!(waits.is_empty());
        let error = refused.err().ok_or("expected invalid credentials")?;
        assert_eq!(super::sqlstate_suffix(&error), " (SQLSTATE 28P01)");
        assert!(
            PostgresPoolError::Postgres(error)
                .to_string()
                .contains("SQLSTATE 28P01")
        );

        let mut attempts = 0;
        let mut waits = Vec::new();
        let exhausted = super::connect_with_retry::<()>(
            || {
                attempts += 1;
                "not a database url".parse::<postgres::Config>().map(|_| ())
            },
            |delay| waits.push(delay),
        );
        assert!(exhausted.is_err());
        assert_eq!(attempts, 3);
        assert_eq!(waits.len(), 2);
        let invalid = "not a database url"
            .parse::<postgres::Config>()
            .err()
            .ok_or("expected invalid configuration")?;
        assert_eq!(super::sqlstate_suffix(&invalid), "");
        assert_eq!(
            super::connect_with_retry(|| Ok::<_, postgres::Error>(7), |_| {})?,
            7
        );
        Ok(())
    }

    #[test]
    fn closed_connections_are_discarded_instead_of_returned()
    -> Result<(), Box<dyn std::error::Error>> {
        test_pool("closed").map_or(Ok(()), |pool| {
            let terminated = backend_pid(&pool)?;
            let mut client = pool.get()?;
            let killer = PostgresPool::shared(&format!("{}_killer", pool.inner.database_url));
            killer
                .get()?
                .execute("SELECT pg_terminate_backend($1, 5000)", &[&terminated])?;
            assert!(client.query_one("SELECT 1", &[]).is_err());
            // A fatal server response can arrive before the driver's closed
            // flag observes EOF. Drive it once more before checking that flag.
            assert!(client.is_valid(Duration::from_secs(1)).is_err());
            assert!(client.is_closed());
            drop(client);

            assert_ne!(backend_pid(&pool)?, terminated);
            Ok(())
        })
    }

    #[test]
    fn idle_connections_are_capped() -> Result<(), Box<dyn std::error::Error>> {
        test_pool("cap").map_or(Ok(()), |pool| {
            let clients = (0..=MAX_IDLE_CONNECTIONS)
                .map(|_| pool.get())
                .collect::<Result<Vec<_>, _>>()?;
            drop(clients);
            let idle = pool.inner.idle.lock().map_or(0, |idle| idle.len());
            assert_eq!(idle, MAX_IDLE_CONNECTIONS);
            Ok(())
        })
    }
}
