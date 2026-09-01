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
CREATE INDEX IF NOT EXISTS idx_credit_ledger_user_settlement_lookup
ON credit_ledger (user_id, (metadata->>'settlement_id'), id DESC)
WHERE metadata ? 'settlement_id';
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
