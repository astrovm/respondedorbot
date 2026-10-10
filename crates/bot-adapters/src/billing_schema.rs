//! Additive PostgreSQL billing schema and data migrations.

use bot_core::credit_units::CREDIT_SCALE;
use postgres::{Client, Transaction};
use serde::Serialize;
use serde_json::json;
use thiserror::Error;

use crate::postgres_connection::postgres_tls_connector;

const CREDIT_UNITS_MIGRATION_ADVISORY_LOCK_KEY: i64 = 48_610_002;
const CREDIT_UNITS_MIGRATION_NAME: &str = "credit_amounts_scaled_to_tenths_v1";
const CREDIT_HUNDREDTHS_MIGRATION_ADVISORY_LOCK_KEY: i64 = 48_610_003;
const CREDIT_HUNDREDTHS_MIGRATION_NAME: &str = "credit_amounts_scaled_to_hundredths_v2";
const COMPACTION_REPAIR_ADVISORY_LOCK_KEY: i64 = 48_610_004;
const COMPACTION_REPAIR_MIGRATION_NAME: &str = "repair_duplicate_compaction_refunds_v1";
const BILLING_SCHEMA_ADVISORY_LOCK_KEY: i64 = 48_610_005;
const LEGACY_WHOLE_TO_TENTHS_FACTOR: i32 = 10;
const TENTHS_TO_HUNDREDTHS_FACTOR: i32 = CREDIT_SCALE as i32 / 10;

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS credit_accounts (
    scope_type TEXT NOT NULL CHECK (scope_type IN ('user', 'chat')),
    scope_id BIGINT NOT NULL,
    balance INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (scope_type, scope_id)
);
CREATE TABLE IF NOT EXISTS onboarding_grants (
    user_id BIGINT PRIMARY KEY,
    credits INTEGER NOT NULL,
    granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS star_payments (
    telegram_payment_charge_id TEXT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    pack_id TEXT NOT NULL,
    xtr_amount INTEGER NOT NULL,
    credits_awarded INTEGER NOT NULL,
    payload TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS lightning_charges (
    charge_id TEXT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    chat_id BIGINT NOT NULL,
    message_id BIGINT,
    pack_id TEXT NOT NULL,
    usd_cents INTEGER NOT NULL,
    credits_awarded INTEGER NOT NULL,
    locale TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'unpaid' CHECK (status IN ('unpaid', 'paid', 'expired')),
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    settled_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_lightning_charges_unpaid
ON lightning_charges (created_at)
WHERE status = 'unpaid';
CREATE TABLE IF NOT EXISTS credit_ledger (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    actor_user_id BIGINT,
    user_id BIGINT,
    chat_id BIGINT,
    amount INTEGER NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_credit_ledger_compaction_usage_tag
ON credit_ledger ((metadata->>'usage_tag'))
WHERE event_type = 'memory_compaction_settlement';
CREATE INDEX IF NOT EXISTS idx_credit_ledger_user_ai_settlements
ON credit_ledger (user_id, created_at DESC, id DESC)
WHERE event_type = 'ai_settlement_result';
CREATE UNIQUE INDEX IF NOT EXISTS idx_credit_ledger_unique_ai_settlement
ON credit_ledger (user_id, (metadata->>'settlement_id'))
WHERE event_type = 'ai_settlement_result' AND metadata ? 'settlement_id';
CREATE INDEX IF NOT EXISTS idx_credit_ledger_settlement_id
ON credit_ledger ((metadata->>'settlement_id'))
WHERE metadata ? 'settlement_id';
CREATE UNIQUE INDEX IF NOT EXISTS idx_credit_ledger_unique_ai_provider_segment
ON credit_ledger ((metadata->>'operation_id'), (metadata->>'segment_id'))
WHERE event_type = 'ai_provider_usage'
  AND metadata ? 'operation_id' AND metadata ? 'segment_id';
CREATE INDEX IF NOT EXISTS idx_credit_ledger_user_charge_history
ON credit_ledger (user_id, id DESC)
WHERE event_type IN (
    'ai_settlement_result', 'memory_compaction_settlement', 'ai_reserve'
);
CREATE INDEX IF NOT EXISTS idx_credit_ledger_user_charge_operations
ON credit_ledger (user_id, id DESC)
WHERE event_type IN (
    'ai_settlement_result', 'memory_compaction_settlement', 'ai_reserve',
    'ai_refund', 'ai_settlement_charge', 'ai_settlement_debt'
);
CREATE INDEX IF NOT EXISTS idx_credit_ledger_chat_created
ON credit_ledger (chat_id, created_at)
WHERE chat_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_credit_ledger_user_settlement_lookup
ON credit_ledger (user_id, (metadata->>'settlement_id'), id DESC)
WHERE metadata ? 'settlement_id';
CREATE UNIQUE INDEX IF NOT EXISTS idx_credit_ledger_unique_command_operation
ON credit_ledger ((metadata->>'command_operation_id'), (metadata->>'direction'))
WHERE metadata ? 'command_operation_id';
CREATE TABLE IF NOT EXISTS credit_schema_migrations (
    name TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
";

#[derive(Debug, Error)]
pub enum BillingSchemaError {
    #[error("could not initialize PostgreSQL TLS: {0}")]
    Tls(#[from] native_tls::Error),
    #[error("PostgreSQL billing schema migration failed: {0}")]
    Postgres(#[from] postgres::Error),
    #[error("billing balance exceeds the PostgreSQL integer range")]
    BalanceOverflow,
    #[error("chat-funded compaction correction requires chat_id")]
    ChatIdRequired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BillingSchemaResult {
    pub migrated_to_tenths: bool,
    pub migrated_to_hundredths: bool,
    pub repaired_compaction_refunds: u64,
}

pub struct BillingSchemaRepository {
    database_url: String,
}

impl BillingSchemaRepository {
    #[must_use]
    pub fn new(database_url: &str) -> Self {
        Self {
            database_url: database_url.to_owned(),
        }
    }

    pub fn ensure_schema(&self) -> Result<BillingSchemaResult, BillingSchemaError> {
        let mut client = self.connect()?;
        let mut transaction = client.transaction()?;
        transaction.query_one(
            "SELECT pg_advisory_xact_lock($1)",
            &[&BILLING_SCHEMA_ADVISORY_LOCK_KEY],
        )?;
        transaction.batch_execute(SCHEMA_SQL)?;
        let migrated_to_tenths = migrate_credit_amounts_to_tenths(&mut transaction)?;
        let migrated_to_hundredths = migrate_credit_amounts_to_hundredths(&mut transaction)?;
        let repaired_compaction_refunds = repair_duplicate_compaction_refunds(&mut transaction)?;
        transaction.commit()?;
        Ok(BillingSchemaResult {
            migrated_to_tenths,
            migrated_to_hundredths,
            repaired_compaction_refunds,
        })
    }

    fn connect(&self) -> Result<Client, BillingSchemaError> {
        Ok(Client::connect(
            &self.database_url,
            postgres_tls_connector(&self.database_url)?,
        )?)
    }
}

fn claim_migration(
    transaction: &mut Transaction<'_>,
    advisory_lock_key: i64,
    name: &str,
) -> Result<bool, BillingSchemaError> {
    transaction.query_one("SELECT pg_advisory_xact_lock($1)", &[&advisory_lock_key])?;
    Ok(transaction
        .query_opt(
            "INSERT INTO credit_schema_migrations (name) VALUES ($1) \
             ON CONFLICT (name) DO NOTHING RETURNING name",
            &[&name],
        )?
        .is_some())
}

fn migrate_credit_amounts_to_tenths(
    transaction: &mut Transaction<'_>,
) -> Result<bool, BillingSchemaError> {
    if !claim_migration(
        transaction,
        CREDIT_UNITS_MIGRATION_ADVISORY_LOCK_KEY,
        CREDIT_UNITS_MIGRATION_NAME,
    )? {
        return Ok(false);
    }
    transaction.execute(
        "UPDATE credit_accounts SET balance = balance * $1",
        &[&LEGACY_WHOLE_TO_TENTHS_FACTOR],
    )?;
    transaction.execute(
        "UPDATE onboarding_grants SET credits = credits * $1",
        &[&LEGACY_WHOLE_TO_TENTHS_FACTOR],
    )?;
    transaction.execute(
        "UPDATE star_payments SET credits_awarded = credits_awarded * $1",
        &[&LEGACY_WHOLE_TO_TENTHS_FACTOR],
    )?;
    transaction.execute(
        "UPDATE credit_ledger SET amount = amount * $1",
        &[&LEGACY_WHOLE_TO_TENTHS_FACTOR],
    )?;
    Ok(true)
}

fn migrate_credit_amounts_to_hundredths(
    transaction: &mut Transaction<'_>,
) -> Result<bool, BillingSchemaError> {
    if !claim_migration(
        transaction,
        CREDIT_HUNDREDTHS_MIGRATION_ADVISORY_LOCK_KEY,
        CREDIT_HUNDREDTHS_MIGRATION_NAME,
    )? {
        return Ok(false);
    }
    transaction.execute(
        "UPDATE credit_accounts SET balance = balance * $1",
        &[&TENTHS_TO_HUNDREDTHS_FACTOR],
    )?;
    transaction.execute(
        "UPDATE onboarding_grants SET credits = credits * $1",
        &[&TENTHS_TO_HUNDREDTHS_FACTOR],
    )?;
    transaction.execute(
        "UPDATE star_payments SET credits_awarded = credits_awarded * $1",
        &[&TENTHS_TO_HUNDREDTHS_FACTOR],
    )?;
    transaction.execute(
        "UPDATE credit_ledger SET amount = amount * $1",
        &[&TENTHS_TO_HUNDREDTHS_FACTOR],
    )?;
    let metadata_factor = i64::from(TENTHS_TO_HUNDREDTHS_FACTOR);
    transaction.execute(
        "UPDATE credit_ledger SET metadata = ( \
            SELECT jsonb_object_agg(item.key, CASE \
                WHEN item.key LIKE '%credit_units%' \
                     AND jsonb_typeof(item.value) = 'number' \
                    THEN to_jsonb((item.value #>> '{}')::bigint * $1) \
                WHEN item.key = ANY(ARRAY[ \
                    'reserved_credits', 'reserved_credits_total', \
                    'settled_credits', 'refunded_credits', \
                    'extra_charged_credits', 'debt_applied_credits' \
                ]::text[]) AND jsonb_typeof(item.value) = 'number' \
                    THEN to_jsonb((item.value #>> '{}')::bigint * $1) \
                ELSE item.value END) \
            FROM jsonb_each(credit_ledger.metadata) AS item \
        ) || jsonb_build_object('credit_scale', $2::bigint) \
        WHERE EXISTS ( \
            SELECT 1 FROM jsonb_each(credit_ledger.metadata) AS item \
            WHERE jsonb_typeof(item.value) = 'number' AND ( \
                item.key LIKE '%credit_units%' OR item.key = ANY(ARRAY[ \
                    'reserved_credits', 'reserved_credits_total', \
                    'settled_credits', 'refunded_credits', \
                    'extra_charged_credits', 'debt_applied_credits' \
                ]::text[]) \
            ) \
        )",
        &[&metadata_factor, &CREDIT_SCALE],
    )?;
    Ok(true)
}

fn repair_duplicate_compaction_refunds(
    transaction: &mut Transaction<'_>,
) -> Result<u64, BillingSchemaError> {
    if !claim_migration(
        transaction,
        COMPACTION_REPAIR_ADVISORY_LOCK_KEY,
        COMPACTION_REPAIR_MIGRATION_NAME,
    )? {
        return Ok(0);
    }
    let repairs = transaction.query(
        "SELECT DISTINCT ON (refund.id) refund.id, refund.user_id, \
            refund.chat_id, refund.amount, refund.metadata->>'source', \
            refund.metadata->>'operation_id', refund.metadata->>'usage_tag' \
         FROM credit_ledger AS refund \
         JOIN credit_ledger AS result ON result.user_id = refund.user_id \
          AND result.event_type = 'ai_settlement_result' \
          AND result.metadata->>'operation_id' = refund.metadata->>'operation_id' \
         JOIN credit_ledger AS legacy ON legacy.user_id = refund.user_id \
          AND legacy.event_type = 'memory_compaction_settlement' \
          AND legacy.metadata->>'usage_tag' = refund.metadata->>'usage_tag' \
          AND legacy.id < result.id \
         WHERE refund.event_type = 'ai_refund' AND refund.amount > 0 \
          AND refund.metadata->>'reason' = 'unused_stale_reservation' \
          AND refund.metadata->>'usage_tag' LIKE 'memory_compaction:%' \
          AND COALESCE(result.metadata->>'settled_credit_units', '0') = '0' \
         ORDER BY refund.id, legacy.id DESC",
        &[],
    )?;
    for row in &repairs {
        let refund_id: i64 = row.try_get(0)?;
        let user_id: i64 = row.try_get(1)?;
        let chat_id: Option<i64> = row.try_get(2)?;
        let amount: i32 = row.try_get(3)?;
        let source = if row.try_get::<_, Option<String>>(4)?.as_deref() == Some("chat") {
            "chat"
        } else {
            "user"
        };
        let operation_id = row.try_get::<_, Option<String>>(5)?.unwrap_or_default();
        let usage_tag = row.try_get::<_, Option<String>>(6)?.unwrap_or_default();
        let user_balance = balance_for_update(transaction, "user", user_id)?;
        let chat_balance = match chat_id {
            Some(value) => Some(balance_for_update(transaction, "chat", value)?),
            None => None,
        };
        if source == "chat" {
            let chat_id = chat_id.ok_or(BillingSchemaError::ChatIdRequired)?;
            let balance = chat_balance
                .ok_or(BillingSchemaError::ChatIdRequired)?
                .checked_sub(amount)
                .ok_or(BillingSchemaError::BalanceOverflow)?;
            set_balance(transaction, "chat", chat_id, balance)?;
        } else {
            let balance = user_balance
                .checked_sub(amount)
                .ok_or(BillingSchemaError::BalanceOverflow)?;
            set_balance(transaction, "user", user_id, balance)?;
        }
        let metadata = json!({
            "source": source,
            "operation_id": operation_id,
            "usage_tag": usage_tag,
            "reversed_refund_ledger_id": refund_id,
            "reason": "duplicate_compaction_refund",
        });
        let correction_amount = amount
            .checked_neg()
            .ok_or(BillingSchemaError::BalanceOverflow)?;
        transaction.execute(
            "INSERT INTO credit_ledger (event_type, actor_user_id, user_id, \
                chat_id, amount, metadata) \
             VALUES ('ai_reconciliation_correction', $1, $1, $2, $3, $4)",
            &[&user_id, &chat_id, &correction_amount, &metadata],
        )?;
    }
    Ok(repairs.len() as u64)
}

fn balance_for_update(
    transaction: &mut Transaction<'_>,
    scope_type: &str,
    scope_id: i64,
) -> Result<i32, BillingSchemaError> {
    transaction.execute(
        "INSERT INTO credit_accounts (scope_type, scope_id, balance) \
         VALUES ($1, $2, 0) ON CONFLICT (scope_type, scope_id) DO NOTHING",
        &[&scope_type, &scope_id],
    )?;
    Ok(transaction
        .query_one(
            "SELECT balance FROM credit_accounts \
             WHERE scope_type = $1 AND scope_id = $2 FOR UPDATE",
            &[&scope_type, &scope_id],
        )?
        .get(0))
}

fn set_balance(
    transaction: &mut Transaction<'_>,
    scope_type: &str,
    scope_id: i64,
    balance: i32,
) -> Result<(), BillingSchemaError> {
    transaction.execute(
        "UPDATE credit_accounts SET balance = $1, updated_at = NOW() \
         WHERE scope_type = $2 AND scope_id = $3",
        &[&balance, &scope_type, &scope_id],
    )?;
    Ok(())
}

/// Test-only PostgreSQL fault injection for billing repositories.
///
/// [`install`](fault_injection::install) wraps chosen tables in views gated by
/// `billing_fault_gate()`, adds statement-level triggers that call the same
/// gate, and shadows the `jsonb ->> text` operator for connections whose
/// `search_path` lists the schema before `pg_catalog`. The gate raises while
/// the running statement's SQL contains a fragment registered with
/// [`inject`](fault_injection::inject), so a test can fail one specific
/// statement in the middle of a real transaction.
#[cfg(test)]
pub(crate) mod fault_injection {
    use std::error::Error;

    use postgres::Client;

    use crate::postgres_connection::postgres_tls_connector;

    pub(crate) type TestResult<T = ()> = Result<T, Box<dyn Error + Send + Sync>>;

    const HARNESS_SQL: &str = "
CREATE TABLE billing_faults (
    fragment TEXT NOT NULL,
    skip_hits BIGINT NOT NULL,
    max_hits BIGINT,
    sql_state TEXT NOT NULL
);
CREATE SEQUENCE billing_fault_hits;
CREATE FUNCTION billing_fault_gate() RETURNS BOOLEAN LANGUAGE plpgsql STABLE AS $$
DECLARE
    fault RECORD;
    hit BIGINT;
BEGIN
    FOR fault IN SELECT * FROM billing_faults LOOP
        IF strpos(current_query(), fault.fragment) > 0 THEN
            hit := nextval('billing_fault_hits');
            IF hit > fault.skip_hits
               AND (fault.max_hits IS NULL OR hit <= fault.skip_hits + fault.max_hits) THEN
                RAISE EXCEPTION USING ERRCODE = fault.sql_state,
                    MESSAGE = 'injected billing fault: ' || fault.fragment;
            END IF;
        END IF;
    END LOOP;
    RETURN TRUE;
END $$;
CREATE FUNCTION billing_fault_trigger() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    PERFORM billing_fault_gate();
    RETURN NULL;
END $$;
CREATE FUNCTION billing_fault_json_text(document JSONB, field TEXT)
RETURNS TEXT LANGUAGE plpgsql IMMUTABLE AS $$
BEGIN
    PERFORM billing_fault_gate();
    RETURN pg_catalog.jsonb_object_field_text(document, field);
END $$;
CREATE OPERATOR ->> (LEFTARG = JSONB, RIGHTARG = TEXT, FUNCTION = billing_fault_json_text);
";

    pub(crate) fn connect(database_url: &str) -> TestResult<Client> {
        let connector = postgres_tls_connector(database_url)?;
        Ok(Client::connect(database_url, connector)?)
    }

    /// Recreates `schema` and returns a URL whose unqualified names resolve
    /// there first, followed by `pg_catalog` and any extra `-c` options.
    pub(crate) fn isolated_schema_url(
        database_url: &str,
        schema: &str,
        extra_options: &str,
    ) -> TestResult<String> {
        let reset = format!("DROP SCHEMA IF EXISTS {schema} CASCADE; CREATE SCHEMA {schema}");
        connect(database_url)?.batch_execute(&reset)?;
        let separator = if database_url.contains('?') { '&' } else { '?' };
        Ok(format!(
            "{database_url}{separator}options=-csearch_path%3D{schema}%2Cpg_catalog{extra_options}"
        ))
    }

    /// Installs the gate; `viewed` tables become gated views over
    /// `<table>_data`, and `triggered` tables gate every write statement.
    pub(crate) fn install(client: &mut Client, viewed: &[&str], triggered: &[&str]) -> TestResult {
        client.batch_execute(HARNESS_SQL)?;
        for table in viewed {
            let wrap = format!(
                "ALTER TABLE {table} RENAME TO {table}_data; \
                 CREATE VIEW {table} AS SELECT * FROM {table}_data WHERE billing_fault_gate(); \
                 CREATE TRIGGER billing_fault BEFORE INSERT ON {table}_data \
                 FOR EACH STATEMENT EXECUTE FUNCTION billing_fault_trigger()"
            );
            client.batch_execute(&wrap)?;
        }
        for table in triggered {
            let gate = format!(
                "CREATE TRIGGER billing_fault BEFORE INSERT OR UPDATE OR DELETE ON {table} \
                 FOR EACH STATEMENT EXECUTE FUNCTION billing_fault_trigger()"
            );
            client.batch_execute(&gate)?;
        }
        Ok(())
    }

    /// Replaces the active fault: statements containing `fragment` fail with
    /// `sql_state` after `skip_hits` matches, at most `max_hits` times.
    pub(crate) fn inject(
        client: &mut Client,
        fragment: &str,
        skip_hits: i64,
        max_hits: Option<i64>,
        sql_state: &str,
    ) -> TestResult {
        clear(client)?;
        let insert = "INSERT INTO billing_faults VALUES ($1, $2, $3, $4)";
        client.execute(insert, &[&fragment, &skip_hits, &max_hits, &sql_state])?;
        Ok(())
    }

    pub(crate) fn clear(client: &mut Client) -> TestResult {
        let reset = "TRUNCATE billing_faults; SELECT setval('billing_fault_hits', 1, false)";
        Ok(client.batch_execute(reset)?)
    }

    /// How many statements matched the active fault's fragment.
    pub(crate) fn hits(client: &mut Client) -> TestResult<i64> {
        let count = "SELECT CASE WHEN is_called THEN last_value ELSE 0 END FROM billing_fault_hits";
        Ok(client.query_one(count, &[])?.get(0))
    }
}

#[cfg(test)]
mod tests {
    use postgres::Client;
    use postgres::error::SqlState;

    use super::fault_injection::{self, TestResult};
    use super::{BillingSchemaError, BillingSchemaRepository, BillingSchemaResult};

    const TENTHS: &str = "credit_amounts_scaled_to_tenths_v1";
    const HUNDREDTHS: &str = "credit_amounts_scaled_to_hundredths_v2";
    const REPAIR: &str = "repair_duplicate_compaction_refunds_v1";

    #[test]
    fn unreachable_database_fails_before_migrating() {
        let result = BillingSchemaRepository::new(
            "postgresql://synthetic@127.0.0.1:1/synthetic?sslmode=disable&connect_timeout=2",
        )
        .ensure_schema();
        assert!(matches!(&result, Err(BillingSchemaError::Postgres(_))));
        assert!(result.is_err_and(|error| {
            error
                .to_string()
                .starts_with("PostgreSQL billing schema migration failed")
        }));
    }

    /// Migration names, total balance, and ledger size: everything a failed
    /// migration must leave untouched.
    fn snapshot(client: &mut Client) -> TestResult<(Option<String>, Option<i64>, i64)> {
        let row = client.query_one(
            "SELECT (SELECT string_agg(name, ',' ORDER BY name COLLATE \"C\") FROM credit_schema_migrations), \
                (SELECT SUM(balance) FROM credit_accounts_data), \
                (SELECT COUNT(*) FROM credit_ledger)",
            &[],
        );
        let row = row?;
        Ok((row.get(0), row.get(1), row.get(2)))
    }

    fn mark_applied(client: &mut Client, names: &[&str]) -> TestResult {
        client.execute("DELETE FROM credit_schema_migrations", &[])?;
        let insert = "INSERT INTO credit_schema_migrations (name) SELECT unnest($1::text[])";
        client.execute(insert, &[&names])?;
        Ok(())
    }

    /// Runs the migration with `fragment` failing and checks that the
    /// transaction rolled back after exactly `skip_hits + 1` matches.
    fn assert_migration_fault(
        client: &mut Client,
        repository: &BillingSchemaRepository,
        applied: &[&str],
        fragment: &str,
        skip_hits: i64,
    ) -> TestResult {
        mark_applied(client, applied)?;
        fault_injection::inject(client, fragment, skip_hits, None, "P0001")?;
        let before = snapshot(client)?;
        let outcome = repository.ensure_schema();
        let expected = format!("injected billing fault: {fragment}");
        assert!(
            matches!(&outcome, Err(BillingSchemaError::Postgres(error))
                if error.as_db_error().is_some_and(|db| db.message() == expected)),
            "{fragment}: {outcome:?}"
        );
        assert_eq!(fault_injection::hits(client)?, skip_hits + 1, "{fragment}");
        assert_eq!(snapshot(client)?, before, "{fragment} must roll back");
        fault_injection::clear(client)?;
        Ok(())
    }

    #[test]
    fn every_migration_statement_failure_rolls_back_the_whole_migration() -> TestResult {
        std::env::var("TEST_DATABASE_URL").map_or(Ok(()), |url| migration_faults(&url))
    }

    fn migration_faults(database_url: &str) -> TestResult {
        let url = fault_injection::isolated_schema_url(database_url, "billing_schema_faults", "")?;
        let repository = BillingSchemaRepository::new(&url);
        assert_eq!(
            repository.ensure_schema()?,
            BillingSchemaResult {
                migrated_to_tenths: true,
                migrated_to_hundredths: true,
                repaired_compaction_refunds: 0,
            }
        );
        let mut client = fault_injection::connect(&url)?;
        let triggered = [
            "onboarding_grants",
            "star_payments",
            "credit_ledger",
            "credit_schema_migrations",
        ];
        fault_injection::install(&mut client, &["credit_accounts"], &triggered)?;
        let seed = "INSERT INTO credit_accounts (scope_type, scope_id, balance) \
                VALUES ('user', 41, 900), ('chat', -42, 800); \
             INSERT INTO credit_ledger \
                (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
             VALUES \
                ('memory_compaction_settlement', 41, 41, -42, 0, \
                    '{\"usage_tag\":\"memory_compaction:fault\"}'), \
                ('ai_settlement_result', 41, 41, -42, 0, \
                    '{\"operation_id\":\"fault-repair\",\"settled_credit_units\":0}'), \
                ('ai_refund', 41, 41, -42, 25, \
                    '{\"source\":\"chat\",\"operation_id\":\"fault-repair\",\
                      \"usage_tag\":\"memory_compaction:fault\",\
                      \"reason\":\"unused_stale_reservation\"}')";
        client.batch_execute(seed)?;

        for (fragment, skip_hits) in [
            ("INSERT INTO credit_schema_migrations", 0),
            ("INSERT INTO credit_schema_migrations", 1),
            ("INSERT INTO credit_schema_migrations", 2),
            ("UPDATE credit_accounts SET balance = balance", 0),
            ("UPDATE onboarding_grants SET credits", 0),
            ("UPDATE star_payments SET credits_awarded", 0),
            ("UPDATE credit_ledger SET amount", 0),
        ] {
            assert_migration_fault(&mut client, &repository, &[], fragment, skip_hits)?;
        }
        for fragment in [
            "UPDATE credit_accounts SET balance = balance",
            "UPDATE onboarding_grants SET credits",
            "UPDATE star_payments SET credits_awarded",
            "UPDATE credit_ledger SET amount",
            "UPDATE credit_ledger SET metadata",
        ] {
            assert_migration_fault(&mut client, &repository, &[TENTHS], fragment, 0)?;
        }
        let migrated = [TENTHS, HUNDREDTHS];
        for fragment in [
            "SELECT DISTINCT ON (refund.id)",
            "VALUES ($1, $2, 0) ON CONFLICT",
            "FOR UPDATE",
            "UPDATE credit_accounts SET balance = $1",
            "'ai_reconciliation_correction'",
        ] {
            assert_migration_fault(&mut client, &repository, &migrated, fragment, 0)?;
        }

        // A migration already holding the schema lock makes this one give up
        // after the configured lock timeout instead of waiting forever.
        let impatient = BillingSchemaRepository::new(&format!("{url}%20-clock_timeout%3D100"));
        client.query_one("SELECT pg_advisory_lock(48610005)", &[])?;
        let blocked = impatient.ensure_schema();
        client.query_one("SELECT pg_advisory_unlock(48610005)", &[])?;
        assert!(matches!(&blocked, Err(BillingSchemaError::Postgres(error))
            if error.code() == Some(&SqlState::LOCK_NOT_AVAILABLE)));

        // Without faults the repair reverses the duplicate chat refund once.
        mark_applied(&mut client, &migrated)?;
        assert_eq!(
            repository.ensure_schema()?,
            BillingSchemaResult {
                migrated_to_tenths: false,
                migrated_to_hundredths: false,
                repaired_compaction_refunds: 1,
            }
        );
        let repaired = client.query_one(
            "SELECT \
                (SELECT balance FROM credit_accounts WHERE scope_type = 'chat' AND scope_id = -42), \
                (SELECT balance FROM credit_accounts WHERE scope_type = 'user' AND scope_id = 41), \
                (SELECT amount FROM credit_ledger \
                    WHERE event_type = 'ai_reconciliation_correction')",
            &[],
        );
        let repaired = repaired?;
        assert_eq!(repaired.get::<_, i32>(0), 775);
        assert_eq!(repaired.get::<_, i32>(1), 900);
        assert_eq!(repaired.get::<_, i32>(2), -25);
        let applied = snapshot(&mut client)?.0;
        let expected = [HUNDREDTHS, TENTHS, REPAIR].join(",");
        assert_eq!(applied.as_deref(), Some(expected.as_str()));
        Ok(())
    }
}
