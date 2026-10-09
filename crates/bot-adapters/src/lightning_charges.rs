//! PostgreSQL records of Lightning top-up charges and their one-time credit.

use serde_json::json;

use crate::billing_read::{BillingError, BillingRepository, BillingScope};

/// A charge created at the payment provider, waiting to be paid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewLightningCharge {
    pub charge_id: String,
    pub user_id: i64,
    pub chat_id: i64,
    pub pack_id: String,
    pub usd_cents: i32,
    pub credits_awarded: i32,
    pub locale: String,
    pub ttl_minutes: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingLightningCharge {
    pub charge_id: String,
    pub chat_id: i64,
    pub message_id: Option<i64>,
    pub locale: String,
    /// Past its expiry by more than the grace period, so it can be closed
    /// even if the provider never reports it expired.
    pub overdue: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightningSettlement {
    pub credits_awarded: i64,
    pub user_balance: i64,
}

impl BillingRepository {
    pub fn record_lightning_charge(&self, charge: &NewLightningCharge) -> Result<(), BillingError> {
        self.run_transaction(|transaction| {
            transaction.execute(
                "INSERT INTO lightning_charges (\
                    charge_id, user_id, chat_id, pack_id, usd_cents, credits_awarded, \
                    locale, expires_at\
                 ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW() + make_interval(mins => $8)) \
                 ON CONFLICT (charge_id) DO NOTHING",
                &[
                    &charge.charge_id,
                    &charge.user_id,
                    &charge.chat_id,
                    &charge.pack_id,
                    &charge.usd_cents,
                    &charge.credits_awarded,
                    &charge.locale,
                    &charge.ttl_minutes,
                ],
            )?;
            Ok(())
        })
    }

    /// Remembers the invoice message so the payment notice can reply to it.
    pub fn attach_lightning_message(
        &self,
        charge_id: &str,
        message_id: i64,
    ) -> Result<(), BillingError> {
        self.run_transaction(|transaction| {
            transaction.execute(
                "UPDATE lightning_charges SET message_id = $2 WHERE charge_id = $1",
                &[&charge_id, &message_id],
            )?;
            Ok(())
        })
    }

    /// Oldest unpaid charges first.
    pub fn pending_lightning_charges(
        &self,
        limit: i64,
        grace_minutes: i32,
    ) -> Result<Vec<PendingLightningCharge>, BillingError> {
        self.run_transaction(|transaction| {
            Ok(transaction
                .query(
                    "SELECT charge_id, chat_id, message_id, locale, \
                        expires_at + make_interval(mins => $2) < NOW() \
                     FROM lightning_charges WHERE status = 'unpaid' \
                     ORDER BY created_at, charge_id LIMIT $1",
                    &[&limit, &grace_minutes],
                )?
                .iter()
                .map(|row| PendingLightningCharge {
                    charge_id: row.get(0),
                    chat_id: row.get(1),
                    message_id: row.get(2),
                    locale: row.get(3),
                    overdue: row.get(4),
                })
                .collect())
        })
    }

    /// Credits a paid charge once. Returns `None` when the charge is unknown
    /// or another worker already credited it. A charge closed as expired is
    /// still credited if the provider reports it paid.
    pub fn settle_lightning_charge(
        &self,
        charge_id: &str,
    ) -> Result<Option<LightningSettlement>, BillingError> {
        self.run_transaction(|transaction| {
            let Some(row) = transaction.query_opt(
                "UPDATE lightning_charges SET status = 'paid', settled_at = NOW() \
                 WHERE charge_id = $1 AND status <> 'paid' \
                 RETURNING user_id, credits_awarded, pack_id, usd_cents",
                &[&charge_id],
            )?
            else {
                return Ok(None);
            };
            let user_id: i64 = row.get(0);
            let credits_awarded: i32 = row.get(1);
            let pack_id: String = row.get(2);
            let usd_cents: i32 = row.get(3);
            let balance =
                BillingRepository::balance_for_update(transaction, BillingScope::User, user_id)?
                    .checked_add(credits_awarded)
                    .ok_or(BillingError::BalanceOverflow)?;
            BillingRepository::set_balance(transaction, BillingScope::User, user_id, balance)?;
            let metadata = json!({
                "source": "lightning",
                "charge_id": charge_id,
                "pack_id": pack_id,
                "usd_cents": usd_cents,
            });
            transaction.execute(
                "INSERT INTO credit_ledger \
                    (event_type, actor_user_id, user_id, amount, metadata) \
                 VALUES ('topup', $1, $1, $2, $3)",
                &[&user_id, &credits_awarded, &metadata],
            )?;
            Ok(Some(LightningSettlement {
                credits_awarded: credits_awarded.into(),
                user_balance: balance.into(),
            }))
        })
    }

    /// Stops polling a charge that can no longer be paid.
    pub fn close_lightning_charge(&self, charge_id: &str) -> Result<(), BillingError> {
        self.run_transaction(|transaction| {
            transaction.execute(
                "UPDATE lightning_charges SET status = 'expired' \
                 WHERE charge_id = $1 AND status = 'unpaid'",
                &[&charge_id],
            )?;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::{LightningSettlement, NewLightningCharge, PendingLightningCharge};
    use crate::billing_read::BillingRepository;
    use crate::billing_schema::{BillingSchemaRepository, fault_injection};

    type TestResult = Result<(), Box<dyn Error + Send + Sync>>;

    const USER: i64 = 7_000_000_000_301;

    fn charge(charge_id: &str, ttl_minutes: i32) -> NewLightningCharge {
        NewLightningCharge {
            charge_id: charge_id.to_owned(),
            user_id: USER,
            chat_id: USER,
            pack_id: "p50".to_owned(),
            usd_cents: 33,
            credits_awarded: 5_000,
            locale: "es".to_owned(),
            ttl_minutes,
        }
    }

    #[test]
    fn lightning_charges_credit_once_and_close_when_expired() -> TestResult {
        let Ok(url) = std::env::var("TEST_DATABASE_URL") else {
            return Ok(());
        };
        BillingSchemaRepository::new(&url).ensure_schema()?;
        let mut client = fault_injection::connect(&url)?;
        client.execute("DELETE FROM lightning_charges WHERE user_id = $1", &[&USER])?;
        client.execute("DELETE FROM credit_ledger WHERE user_id = $1", &[&USER])?;
        client.execute(
            "DELETE FROM credit_accounts WHERE scope_type = 'user' AND scope_id = $1",
            &[&USER],
        )?;
        let repository = BillingRepository::new(&url);

        repository.record_lightning_charge(&charge("synthetic-ln-paid", 30))?;
        repository.record_lightning_charge(&charge("synthetic-ln-paid", 30))?;
        repository.record_lightning_charge(&charge("synthetic-ln-old", 10))?;
        client.execute(
            "UPDATE lightning_charges SET expires_at = NOW() - INTERVAL '20 minutes', \
                created_at = NOW() - INTERVAL '1 hour' \
             WHERE charge_id = 'synthetic-ln-old'",
            &[],
        )?;
        repository.attach_lightning_message("synthetic-ln-paid", 55)?;
        let pending = repository.pending_lightning_charges(10, 10)?;
        let ours = pending
            .into_iter()
            .filter(|charge| charge.charge_id.starts_with("synthetic-ln-"))
            .collect::<Vec<_>>();
        assert_eq!(
            ours,
            [
                PendingLightningCharge {
                    charge_id: "synthetic-ln-old".to_owned(),
                    chat_id: USER,
                    message_id: None,
                    locale: "es".to_owned(),
                    overdue: true,
                },
                PendingLightningCharge {
                    charge_id: "synthetic-ln-paid".to_owned(),
                    chat_id: USER,
                    message_id: Some(55),
                    locale: "es".to_owned(),
                    overdue: false,
                },
            ]
        );

        assert_eq!(
            repository.settle_lightning_charge("synthetic-ln-paid")?,
            Some(LightningSettlement {
                credits_awarded: 5_000,
                user_balance: 5_000,
            })
        );
        assert_eq!(
            repository.settle_lightning_charge("synthetic-ln-paid")?,
            None
        );
        assert_eq!(
            repository.settle_lightning_charge("synthetic-ln-missing")?,
            None
        );

        repository.close_lightning_charge("synthetic-ln-old")?;
        repository.close_lightning_charge("synthetic-ln-paid")?;
        assert!(
            repository
                .pending_lightning_charges(1_000, 10)?
                .iter()
                .all(|charge| !charge.charge_id.starts_with("synthetic-ln-"))
        );
        // A late payment on a closed charge is still credited.
        assert_eq!(
            repository.settle_lightning_charge("synthetic-ln-old")?,
            Some(LightningSettlement {
                credits_awarded: 5_000,
                user_balance: 10_000,
            })
        );
        let row = client.query_one(
            "SELECT COUNT(*), COUNT(*) FILTER (WHERE metadata->>'source' = 'lightning' \
                AND metadata->>'usd_cents' = '33' AND metadata->>'pack_id' = 'p50') \
             FROM credit_ledger WHERE user_id = $1 AND event_type = 'topup'",
            &[&USER],
        )?;
        assert_eq!((row.get::<_, i64>(0), row.get::<_, i64>(1)), (2, 2));
        let statuses = client.query(
            "SELECT status FROM lightning_charges WHERE user_id = $1 ORDER BY charge_id",
            &[&USER],
        )?;
        let statuses = statuses
            .iter()
            .map(|row| row.get::<_, String>(0))
            .collect::<Vec<_>>();
        assert_eq!(statuses, ["paid", "paid"]);
        Ok(())
    }

    #[test]
    fn settling_into_a_full_balance_reports_overflow() -> TestResult {
        let Ok(url) = std::env::var("TEST_DATABASE_URL") else {
            return Ok(());
        };
        BillingSchemaRepository::new(&url).ensure_schema()?;
        let user = USER + 1;
        let mut client = fault_injection::connect(&url)?;
        client.execute("DELETE FROM lightning_charges WHERE user_id = $1", &[&user])?;
        client.execute(
            "INSERT INTO credit_accounts (scope_type, scope_id, balance) \
             VALUES ('user', $1, 2147483647) \
             ON CONFLICT (scope_type, scope_id) DO UPDATE SET balance = 2147483647",
            &[&user],
        )?;
        let repository = BillingRepository::new(&url);
        repository.record_lightning_charge(&NewLightningCharge {
            user_id: user,
            ..charge("synthetic-ln-overflow", 30)
        })?;
        assert!(matches!(
            repository.settle_lightning_charge("synthetic-ln-overflow"),
            Err(crate::billing_read::BillingError::BalanceOverflow)
        ));
        // The failed transaction left the charge unpaid.
        let status = client.query_one(
            "SELECT status FROM lightning_charges WHERE charge_id = 'synthetic-ln-overflow'",
            &[],
        )?;
        assert_eq!(status.get::<_, String>(0), "unpaid");
        client.execute("DELETE FROM lightning_charges WHERE user_id = $1", &[&user])?;
        Ok(())
    }
}
