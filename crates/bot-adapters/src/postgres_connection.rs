//! PostgreSQL connection-string validation at the configuration boundary.

use std::str::FromStr;

use native_tls::{Certificate, TlsConnector};
use postgres::Config;
use postgres::config::Host;
use postgres_native_tls::MakeTlsConnector;
use thiserror::Error;

const SUPABASE_POOLER_SUFFIX: &str = ".pooler.supabase.com";
const SUPABASE_TRANSACTION_POOLER_PORT: u16 = 6543;
const SUPABASE_ROOT_CA_PEM: &[u8] = include_bytes!("../certificates/supabase-root-2021-ca.crt");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PostgresConnectionStringError {
    #[error("must be a valid PostgreSQL connection string")]
    Invalid,
    #[error("must use the Supabase session pooler on port 5432, not transaction port 6543")]
    SupabaseTransactionPooler,
}

pub fn validate_connection_string(
    connection_string: &str,
) -> Result<(), PostgresConnectionStringError> {
    let config =
        Config::from_str(connection_string).map_err(|_| PostgresConnectionStringError::Invalid)?;
    if uses_supabase_pooler(&config)
        && config
            .get_ports()
            .contains(&SUPABASE_TRANSACTION_POOLER_PORT)
    {
        return Err(PostgresConnectionStringError::SupabaseTransactionPooler);
    }
    Ok(())
}

/// Builds a PostgreSQL TLS connector, adding Supabase's root only for its pooler.
pub fn postgres_tls_connector(
    connection_string: &str,
) -> Result<MakeTlsConnector, native_tls::Error> {
    let mut builder = TlsConnector::builder();
    if Config::from_str(connection_string).is_ok_and(|config| uses_supabase_pooler(&config)) {
        builder.add_root_certificate(Certificate::from_pem(SUPABASE_ROOT_CA_PEM)?);
    }
    Ok(MakeTlsConnector::new(builder.build()?))
}

fn uses_supabase_pooler(config: &Config) -> bool {
    config.get_hosts().iter().any(
        |host| matches!(host, Host::Tcp(hostname) if hostname.ends_with(SUPABASE_POOLER_SUFFIX)),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        PostgresConnectionStringError, postgres_tls_connector, validate_connection_string,
    };

    #[test]
    fn builds_postgres_tls_connector_with_scoped_supabase_root() {
        for connection_string in [
            "postgresql://synthetic-user:synthetic-password@db.synthetic.invalid:5432/postgres?sslmode=require",
            "postgres://postgres.synthetic-project:synthetic-password@aws-0-synthetic.pooler.supabase.com:5432/postgres?sslmode=require",
        ] {
            assert!(postgres_tls_connector(connection_string).is_ok());
        }
    }

    #[test]
    fn accepts_postgresql_and_supabase_session_connections() {
        for connection_string in [
            "postgresql://synthetic-user:synthetic-password@db.synthetic.invalid:5432/postgres?sslmode=require",
            "postgres://postgres.synthetic-project:synthetic-password@aws-0-synthetic.pooler.supabase.com:5432/postgres?sslmode=require",
        ] {
            assert_eq!(validate_connection_string(connection_string), Ok(()));
        }
    }

    #[test]
    fn rejects_unknown_options_without_exposing_the_connection_string() {
        let secret = "synthetic-database-password";
        let connection_string = format!(
            "postgresql://synthetic-user:{secret}@db.synthetic.invalid:5432/postgres?sslmode=require&supa=unsupported"
        );
        let result = validate_connection_string(&connection_string);
        assert_eq!(result, Err(PostgresConnectionStringError::Invalid));
        assert!(!format!("{result:?}").contains(secret));
    }

    #[test]
    fn rejects_supabase_transaction_pooling_for_the_persistent_runtime() {
        let result = validate_connection_string(
            "postgres://postgres.synthetic-project:synthetic-password@aws-0-synthetic.pooler.supabase.com:6543/postgres?sslmode=require",
        );
        assert_eq!(
            result,
            Err(PostgresConnectionStringError::SupabaseTransactionPooler)
        );
    }
}
