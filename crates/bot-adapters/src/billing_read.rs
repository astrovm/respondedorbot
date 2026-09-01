//! PostgreSQL billing repository.

use postgres::{Client, Transaction, error::SqlState};
use serde::Serialize;
use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::postgres_connection::postgres_tls_connector;

const ONBOARDING_MAX_GRANTS_PER_HOUR: i64 = 4;
const ONBOARDING_MAX_GRANTS_PER_DAY: i64 = 16;
const ONBOARDING_GRANTS_ADVISORY_LOCK_KEY: i64 = 48_610_001;
const CREDIT_TRANSACTION_MAX_ATTEMPTS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BillingScope {
    User,
    Chat,
}

impl BillingScope {
    fn parse(value: &str) -> Result<Self, BillingError> {
        match value {
            "user" => Ok(Self::User),
            "chat" => Ok(Self::Chat),
            _ => Err(BillingError::InvalidScope(value.to_owned())),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Chat => "chat",
        }
    }
}

#[derive(Debug, Error)]
pub enum BillingError {
    #[error("billing scope must be user or chat, got {0}")]
    InvalidScope(String),
    #[error("could not initialize PostgreSQL TLS: {0}")]
    Tls(#[from] native_tls::Error),
    #[error("PostgreSQL billing read failed: {0}")]
    Postgres(#[from] postgres::Error),
    #[error("PostgreSQL onboarding transaction retry limit was exhausted")]
    TransactionRetriesExhausted,
    #[error("billing balance exceeds the PostgreSQL integer range")]
    BalanceOverflow,
    #[error("AI operation has more than one payer")]
    MultiplePayers,
    #[error("chat-funded AI operation requires chat_id")]
    ChatIdRequired,
    #[error("chat-funded settlement requires chat_id")]
    LegacyChatIdRequired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct OnboardingGrantResult {
    pub granted: bool,
    pub balance: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StarPaymentResult {
    pub inserted: bool,
    pub user_balance: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct TransferResult {
    pub transferred: bool,
    pub user_balance: i64,
    pub chat_balance: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ChatAiChargeResult {
    pub charged: bool,
    pub chat_balance: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BalancePairResult {
    pub user_balance: i64,
    pub chat_balance: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AiRefundResult {
    pub applied: bool,
    pub reason: Option<String>,
    pub user_balance: i64,
    pub chat_balance: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AiChargeResult {
    pub ok: bool,
    pub applied: bool,
    pub reason: Option<String>,
    pub source: Option<String>,
    pub amount: i64,
    pub user_balance: i64,
    pub chat_balance: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AiSettlementResult {
    pub applied: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authorized_credit_units: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual_credit_units: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refunded_credit_units: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debt_applied_credit_units: Option<i64>,
    pub user_balance: i64,
    pub chat_balance: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LegacySettlementResult {
    pub applied: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adjustment_credit_units: Option<i64>,
    pub user_balance: i64,
    pub chat_balance: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LedgerEntry {
    pub id: i64,
    pub event_type: String,
    pub actor_user_id: Option<i64>,
    pub user_id: Option<i64>,
    pub chat_id: Option<i64>,
    pub amount: i32,
    pub metadata: Value,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UnsettledAiOperation {
    pub operation_id: String,
    pub user_id: i64,
    pub chat_id: Option<i64>,
    pub authorized_credit_units: i64,
    pub source: String,
    pub created_at: String,
    pub last_activity_at: String,
    pub reserve_metadata: Value,
    pub segments: Vec<Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PurgeResult {
    pub deleted_rows: u64,
    pub retention_days: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChargeHistoryRow {
    pub id: i64,
    pub event_type: String,
    pub actor_user_id: Option<i64>,
    pub user_id: Option<i64>,
    pub chat_id: Option<i64>,
    pub amount: i64,
    pub metadata: Value,
    pub created_at: String,
    pub group_key: String,
    pub group_cursor: i64,
    pub group_created_at: String,
}

pub struct BillingRepository {
    database_url: String,
}

impl BillingRepository {
    pub fn new(database_url: &str) -> Self {
        Self {
            database_url: database_url.to_owned(),
        }
    }

    pub fn get_balance(&self, scope_type: &str, scope_id: i64) -> Result<i64, BillingError> {
        let scope = BillingScope::parse(scope_type)?;
        let mut client = self.connect()?;
        let row = client.query_opt(
            "SELECT balance FROM credit_accounts WHERE scope_type = $1 AND scope_id = $2",
            &[&scope.as_str(), &scope_id],
        )?;
        Ok(row.map_or(0, |value| value.get::<_, i32>(0).into()))
    }

    pub fn get_or_create_balance(
        &self,
        scope_type: &str,
        scope_id: i64,
    ) -> Result<i64, BillingError> {
        let scope = BillingScope::parse(scope_type)?;
        let mut client = self.connect()?;
        let mut transaction = client.transaction()?;
        transaction.execute(
            "INSERT INTO credit_accounts (scope_type, scope_id, balance) VALUES ($1, $2, 0) \
             ON CONFLICT (scope_type, scope_id) DO NOTHING",
            &[&scope.as_str(), &scope_id],
        )?;
        let row = transaction.query_one(
            "SELECT balance FROM credit_accounts WHERE scope_type = $1 AND scope_id = $2",
            &[&scope.as_str(), &scope_id],
        )?;
        let balance = row.get::<_, i32>(0).into();
        transaction.commit()?;
        Ok(balance)
    }

    pub fn grant_onboarding_if_needed(
        &self,
        user_id: i64,
        credits: i32,
    ) -> Result<OnboardingGrantResult, BillingError> {
        self.run_transaction(|transaction| {
            Self::apply_onboarding_grant(transaction, user_id, credits)
        })
    }

    pub fn record_star_payment(
        &self,
        charge_id: &str,
        user_id: i64,
        pack_id: &str,
        xtr_amount: i32,
        credits_awarded: i32,
        payload: Option<&str>,
    ) -> Result<StarPaymentResult, BillingError> {
        self.run_transaction(|transaction| {
            let inserted = transaction
                .query_opt(
                    "INSERT INTO star_payments (\
                        telegram_payment_charge_id, user_id, pack_id, xtr_amount, \
                        credits_awarded, payload\
                     ) VALUES ($1, $2, $3, $4, $5, $6) \
                     ON CONFLICT (telegram_payment_charge_id) DO NOTHING \
                     RETURNING telegram_payment_charge_id",
                    &[
                        &charge_id,
                        &user_id,
                        &pack_id,
                        &xtr_amount,
                        &credits_awarded,
                        &payload,
                    ],
                )?
                .is_some();
            let mut user_balance =
                Self::balance_for_update(transaction, BillingScope::User, user_id)?;
            if inserted {
                user_balance = user_balance
                    .checked_add(credits_awarded)
                    .ok_or(BillingError::BalanceOverflow)?;
                Self::set_balance(transaction, BillingScope::User, user_id, user_balance)?;
                let metadata = json!({
                    "pack_id": pack_id,
                    "xtr_amount": xtr_amount,
                    "charge_id": charge_id,
                });
                transaction.execute(
                    "INSERT INTO credit_ledger \
                        (event_type, actor_user_id, user_id, amount, metadata) \
                     VALUES ('topup', $1, $1, $2, $3)",
                    &[&user_id, &credits_awarded, &metadata],
                )?;
            }
            Ok(StarPaymentResult {
                inserted,
                user_balance: user_balance.into(),
            })
        })
    }

    pub fn mint_user_credits(
        &self,
        user_id: i64,
        amount: i32,
        actor_user_id: Option<i64>,
    ) -> Result<i64, BillingError> {
        self.run_transaction(|transaction| {
            let balance = Self::balance_for_update(transaction, BillingScope::User, user_id)?
                .checked_add(amount)
                .ok_or(BillingError::BalanceOverflow)?;
            Self::set_balance(transaction, BillingScope::User, user_id, balance)?;
            let actor_user_id = actor_user_id.unwrap_or(user_id);
            let metadata = json!({"source": "admin_command"});
            transaction.execute(
                "INSERT INTO credit_ledger \
                    (event_type, actor_user_id, user_id, amount, metadata) \
                 VALUES ('printcredits', $1, $2, $3, $4)",
                &[&actor_user_id, &user_id, &amount, &metadata],
            )?;
            Ok(balance.into())
        })
    }

    pub fn transfer_user_to_chat(
        &self,
        user_id: i64,
        chat_id: i64,
        amount: i32,
    ) -> Result<TransferResult, BillingError> {
        self.run_transaction(|transaction| {
            let user_balance = Self::balance_for_update(transaction, BillingScope::User, user_id)?;
            let chat_balance = Self::balance_for_update(transaction, BillingScope::Chat, chat_id)?;
            if user_balance < amount {
                return Ok(TransferResult {
                    transferred: false,
                    user_balance: user_balance.into(),
                    chat_balance: chat_balance.into(),
                });
            }

            let updated_user_balance = user_balance
                .checked_sub(amount)
                .ok_or(BillingError::BalanceOverflow)?;
            let updated_chat_balance = chat_balance
                .checked_add(amount)
                .ok_or(BillingError::BalanceOverflow)?;
            Self::set_balance(
                transaction,
                BillingScope::User,
                user_id,
                updated_user_balance,
            )?;
            Self::set_balance(
                transaction,
                BillingScope::Chat,
                chat_id,
                updated_chat_balance,
            )?;
            let user_amount = amount.checked_neg().ok_or(BillingError::BalanceOverflow)?;
            let user_metadata = json!({"direction": "user_to_chat"});
            transaction.execute(
                "INSERT INTO credit_ledger \
                    (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
                 VALUES ('transfer_user_to_chat', $1, $1, $2, $3, $4)",
                &[&user_id, &chat_id, &user_amount, &user_metadata],
            )?;
            let chat_metadata = json!({"direction": "chat_from_user"});
            transaction.execute(
                "INSERT INTO credit_ledger \
                    (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
                 VALUES ('transfer_user_to_chat', $1, $1, $2, $3, $4)",
                &[&user_id, &chat_id, &amount, &chat_metadata],
            )?;
            Ok(TransferResult {
                transferred: true,
                user_balance: updated_user_balance.into(),
                chat_balance: updated_chat_balance.into(),
            })
        })
    }

    pub fn charge_chat_ai_credits(
        &self,
        chat_id: i64,
        amount: i32,
        event_type: &str,
        metadata: &Map<String, Value>,
    ) -> Result<ChatAiChargeResult, BillingError> {
        self.run_transaction(|transaction| {
            let chat_balance = Self::balance_for_update(transaction, BillingScope::Chat, chat_id)?;
            if chat_balance < amount {
                return Ok(ChatAiChargeResult {
                    charged: false,
                    chat_balance: chat_balance.into(),
                });
            }
            let updated_balance = chat_balance
                .checked_sub(amount)
                .ok_or(BillingError::BalanceOverflow)?;
            Self::set_balance(transaction, BillingScope::Chat, chat_id, updated_balance)?;
            let ledger_amount = amount.checked_neg().ok_or(BillingError::BalanceOverflow)?;
            let metadata = billing_metadata("chat", metadata);
            transaction.execute(
                "INSERT INTO credit_ledger \
                    (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
                 VALUES ($1, NULL, NULL, $2, $3, $4)",
                &[&event_type, &chat_id, &ledger_amount, &metadata],
            )?;
            Ok(ChatAiChargeResult {
                charged: true,
                chat_balance: updated_balance.into(),
            })
        })
    }

    pub fn refund_chat_ai_credits(
        &self,
        chat_id: i64,
        amount: i32,
        event_type: &str,
        metadata: &Map<String, Value>,
    ) -> Result<i64, BillingError> {
        self.mutate_chat_ai_balance(chat_id, amount, event_type, metadata)
    }

    pub fn apply_chat_ai_debt(
        &self,
        chat_id: i64,
        amount: i32,
        event_type: &str,
        metadata: &Map<String, Value>,
    ) -> Result<i64, BillingError> {
        let balance_delta = amount.checked_neg().ok_or(BillingError::BalanceOverflow)?;
        self.mutate_chat_ai_balance(chat_id, balance_delta, event_type, metadata)
    }

    fn mutate_chat_ai_balance(
        &self,
        chat_id: i64,
        balance_delta: i32,
        event_type: &str,
        metadata: &Map<String, Value>,
    ) -> Result<i64, BillingError> {
        self.run_transaction(|transaction| {
            let chat_balance = Self::balance_for_update(transaction, BillingScope::Chat, chat_id)?;
            let updated_balance = chat_balance
                .checked_add(balance_delta)
                .ok_or(BillingError::BalanceOverflow)?;
            Self::set_balance(transaction, BillingScope::Chat, chat_id, updated_balance)?;
            let metadata = billing_metadata("chat", metadata);
            transaction.execute(
                "INSERT INTO credit_ledger \
                    (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
                 VALUES ($1, NULL, NULL, $2, $3, $4)",
                &[&event_type, &chat_id, &balance_delta, &metadata],
            )?;
            Ok(updated_balance.into())
        })
    }

    pub fn apply_ai_debt(
        &self,
        user_id: i64,
        chat_id: Option<i64>,
        amount: i32,
        source: &str,
        event_type: &str,
        metadata: &Map<String, Value>,
    ) -> Result<BalancePairResult, BillingError> {
        self.run_transaction(|transaction| {
            let user_balance = Self::balance_for_update(transaction, BillingScope::User, user_id)?;
            let chat_balance = match chat_id {
                Some(chat_id) => {
                    Self::balance_for_update(transaction, BillingScope::Chat, chat_id)?
                }
                None => 0,
            };
            let debt_delta = amount.checked_neg().ok_or(BillingError::BalanceOverflow)?;
            let (updated_user_balance, updated_chat_balance, ledger_source) =
                if let Some(payer_chat_id) = chat_id.filter(|_| source == "chat") {
                    let updated_chat_balance = chat_balance
                        .checked_add(debt_delta)
                        .ok_or(BillingError::BalanceOverflow)?;
                    Self::set_balance(
                        transaction,
                        BillingScope::Chat,
                        payer_chat_id,
                        updated_chat_balance,
                    )?;
                    (user_balance, updated_chat_balance, "chat")
                } else {
                    let updated_user_balance = user_balance
                        .checked_add(debt_delta)
                        .ok_or(BillingError::BalanceOverflow)?;
                    Self::set_balance(
                        transaction,
                        BillingScope::User,
                        user_id,
                        updated_user_balance,
                    )?;
                    (updated_user_balance, chat_balance, "user")
                };
            let metadata = billing_metadata(ledger_source, metadata);
            transaction.execute(
                "INSERT INTO credit_ledger \
                    (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
                 VALUES ($1, $2, $2, $3, $4, $5)",
                &[&event_type, &user_id, &chat_id, &debt_delta, &metadata],
            )?;
            Ok(BalancePairResult {
                user_balance: updated_user_balance.into(),
                chat_balance: updated_chat_balance.into(),
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn refund_ai_charge(
        &self,
        user_id: i64,
        chat_id: Option<i64>,
        amount: i32,
        source: &str,
        event_type: &str,
        metadata: &Map<String, Value>,
        idempotency_key: Option<&str>,
        operation_id: &str,
    ) -> Result<AiRefundResult, BillingError> {
        let metadata = ai_mutation_metadata(metadata, idempotency_key, operation_id);
        self.run_transaction(|transaction| {
            let user_balance = Self::balance_for_update(transaction, BillingScope::User, user_id)?;
            let chat_balance = match chat_id {
                Some(chat_id) => {
                    Self::balance_for_update(transaction, BillingScope::Chat, chat_id)?
                }
                None => 0,
            };
            if let Some(idempotency_key) = idempotency_key {
                let duplicate = transaction
                    .query_opt(
                        "SELECT 1 FROM credit_ledger \
                         WHERE user_id = $1 AND event_type = $2 \
                           AND metadata->>'idempotency_key' = $3 LIMIT 1",
                        &[&user_id, &event_type, &idempotency_key],
                    )?
                    .is_some();
                if duplicate {
                    return Ok(AiRefundResult {
                        applied: false,
                        reason: None,
                        user_balance: user_balance.into(),
                        chat_balance: chat_balance.into(),
                    });
                }
            }
            if !operation_id.is_empty()
                && transaction
                    .query_opt(
                        "SELECT 1 FROM credit_ledger \
                         WHERE user_id = $1 AND event_type = 'ai_settlement_result' \
                           AND metadata->>'operation_id' = $2 LIMIT 1",
                        &[&user_id, &operation_id],
                    )?
                    .is_some()
            {
                return Ok(AiRefundResult {
                    applied: false,
                    reason: Some("operation_settled".to_owned()),
                    user_balance: user_balance.into(),
                    chat_balance: chat_balance.into(),
                });
            }

            let (updated_user_balance, updated_chat_balance, ledger_source) =
                if let Some(payer_chat_id) = chat_id.filter(|_| source == "chat") {
                    let updated_chat_balance = chat_balance
                        .checked_add(amount)
                        .ok_or(BillingError::BalanceOverflow)?;
                    Self::set_balance(
                        transaction,
                        BillingScope::Chat,
                        payer_chat_id,
                        updated_chat_balance,
                    )?;
                    (user_balance, updated_chat_balance, "chat")
                } else {
                    let updated_user_balance = user_balance
                        .checked_add(amount)
                        .ok_or(BillingError::BalanceOverflow)?;
                    Self::set_balance(
                        transaction,
                        BillingScope::User,
                        user_id,
                        updated_user_balance,
                    )?;
                    (updated_user_balance, chat_balance, "user")
                };
            let metadata = billing_metadata(ledger_source, &metadata);
            transaction.execute(
                "INSERT INTO credit_ledger \
                    (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
                 VALUES ($1, $2, $2, $3, $4, $5)",
                &[&event_type, &user_id, &chat_id, &amount, &metadata],
            )?;
            Ok(AiRefundResult {
                applied: true,
                reason: None,
                user_balance: updated_user_balance.into(),
                chat_balance: updated_chat_balance.into(),
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn charge_ai_credits(
        &self,
        user_id: i64,
        chat_id: Option<i64>,
        amount: i32,
        event_type: &str,
        metadata: &Map<String, Value>,
        source: Option<&str>,
        idempotency_key: Option<&str>,
        operation_id: &str,
    ) -> Result<AiChargeResult, BillingError> {
        let metadata = ai_mutation_metadata(metadata, idempotency_key, operation_id);
        self.run_transaction(|transaction| {
            let user_balance = Self::balance_for_update(transaction, BillingScope::User, user_id)?;
            let chat_balance = match chat_id {
                Some(chat_id) => {
                    Self::balance_for_update(transaction, BillingScope::Chat, chat_id)?
                }
                None => 0,
            };

            if let Some(idempotency_key) = idempotency_key
                && let Some(existing) = transaction.query_opt(
                    "SELECT amount, metadata->>'source' FROM credit_ledger \
                     WHERE user_id = $1 AND event_type = $2 \
                       AND metadata->>'idempotency_key' = $3 \
                     ORDER BY id DESC LIMIT 1",
                    &[&user_id, &event_type, &idempotency_key],
                )?
            {
                if event_type == "ai_reserve" {
                    let reservation_refunded = transaction
                        .query_opt(
                            "SELECT 1 FROM credit_ledger \
                                 WHERE user_id = $1 AND event_type = 'ai_refund' \
                                   AND metadata->>'settlement_id' = $2 LIMIT 1",
                            &[&user_id, &idempotency_key],
                        )?
                        .is_some();
                    if reservation_refunded {
                        return Ok(Self::rejected_ai_charge(
                            "reservation_refunded",
                            user_balance,
                            chat_balance,
                        ));
                    }
                    if Self::ai_operation_is_settled(transaction, user_id, operation_id)? {
                        return Ok(Self::rejected_ai_charge(
                            "operation_settled",
                            user_balance,
                            chat_balance,
                        ));
                    }
                }
                let existing_amount = existing.get::<_, i32>(0);
                let existing_source = existing
                    .get::<_, Option<String>>(1)
                    .filter(|value| !value.is_empty())
                    .unwrap_or_else(|| "user".to_owned());
                return Ok(AiChargeResult {
                    ok: true,
                    applied: false,
                    reason: None,
                    source: Some(existing_source),
                    amount: (-i64::from(existing_amount)).max(0),
                    user_balance: user_balance.into(),
                    chat_balance: chat_balance.into(),
                });
            }

            if event_type == "ai_reserve"
                && Self::ai_operation_is_settled(transaction, user_id, operation_id)?
            {
                return Ok(Self::rejected_ai_charge(
                    "operation_settled",
                    user_balance,
                    chat_balance,
                ));
            }

            let ledger_amount = amount.checked_neg().ok_or(BillingError::BalanceOverflow)?;
            if source != Some("chat") && user_balance >= amount {
                let updated_user_balance = user_balance
                    .checked_sub(amount)
                    .ok_or(BillingError::BalanceOverflow)?;
                Self::set_balance(
                    transaction,
                    BillingScope::User,
                    user_id,
                    updated_user_balance,
                )?;
                Self::insert_ai_charge_ledger(
                    transaction,
                    event_type,
                    user_id,
                    chat_id,
                    ledger_amount,
                    "user",
                    &metadata,
                )?;
                return Ok(AiChargeResult {
                    ok: true,
                    applied: true,
                    reason: None,
                    source: Some("user".to_owned()),
                    amount: amount.into(),
                    user_balance: updated_user_balance.into(),
                    chat_balance: chat_balance.into(),
                });
            }
            if source != Some("user")
                && let Some(payer_chat_id) = chat_id
                && chat_balance >= amount
            {
                let updated_chat_balance = chat_balance
                    .checked_sub(amount)
                    .ok_or(BillingError::BalanceOverflow)?;
                Self::set_balance(
                    transaction,
                    BillingScope::Chat,
                    payer_chat_id,
                    updated_chat_balance,
                )?;
                Self::insert_ai_charge_ledger(
                    transaction,
                    event_type,
                    user_id,
                    chat_id,
                    ledger_amount,
                    "chat",
                    &metadata,
                )?;
                return Ok(AiChargeResult {
                    ok: true,
                    applied: true,
                    reason: None,
                    source: Some("chat".to_owned()),
                    amount: amount.into(),
                    user_balance: user_balance.into(),
                    chat_balance: updated_chat_balance.into(),
                });
            }
            Ok(AiChargeResult {
                ok: false,
                applied: false,
                reason: None,
                source: None,
                amount: 0,
                user_balance: user_balance.into(),
                chat_balance: chat_balance.into(),
            })
        })
    }

    pub fn record_ai_provider_usage(
        &self,
        user_id: i64,
        chat_id: Option<i64>,
        metadata: &Value,
    ) -> Result<bool, BillingError> {
        self.run_transaction(|transaction| {
            Ok(transaction
                .query_opt(
                    "INSERT INTO credit_ledger \
                        (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
                     VALUES ('ai_provider_usage', $1, $1, $2, 0, $3) \
                     ON CONFLICT DO NOTHING RETURNING id",
                    &[&user_id, &chat_id, &metadata],
                )?
                .is_some())
        })
    }

    pub fn list_ai_provider_segments(
        &self,
        user_id: i64,
        operation_id: &str,
    ) -> Result<Vec<Value>, BillingError> {
        let mut client = self.connect()?;
        Ok(client
            .query(
                "SELECT metadata->'segment' FROM credit_ledger \
                 WHERE user_id = $1 AND event_type = 'ai_provider_usage' \
                   AND metadata->>'operation_id' = $2 ORDER BY id",
                &[&user_id, &operation_id],
            )?
            .into_iter()
            .map(|row| row.get(0))
            .filter(Value::is_object)
            .collect())
    }

    pub fn update_ai_provider_usage(
        &self,
        operation_id: &str,
        segment_id: &str,
        segment: &Value,
    ) -> Result<bool, BillingError> {
        self.run_transaction(|transaction| {
            Ok(transaction
                .query_opt(
                    "UPDATE credit_ledger \
                     SET metadata = jsonb_set(metadata, '{segment}', $1) \
                     WHERE event_type = 'ai_provider_usage' \
                       AND metadata->>'operation_id' = $2 \
                       AND metadata->>'segment_id' = $3 RETURNING id",
                    &[&segment, &operation_id, &segment_id],
                )?
                .is_some())
        })
    }

    pub fn compaction_reservation_settled(
        &self,
        user_id: i64,
        operation_id: &str,
        usage_tag: &str,
    ) -> Result<bool, BillingError> {
        let mut client = self.connect()?;
        Ok(client
            .query_opt(
                "SELECT 1 FROM credit_ledger WHERE user_id = $1 AND ( \
                    (event_type = 'ai_settlement_result' \
                        AND NULLIF($2, '') IS NOT NULL \
                        AND metadata->>'operation_id' = $2) \
                    OR (event_type = 'memory_compaction_settlement' \
                        AND NULLIF($3, '') IS NOT NULL \
                        AND metadata->>'usage_tag' = $3) \
                 ) LIMIT 1",
                &[&user_id, &operation_id, &usage_tag],
            )?
            .is_some())
    }

    pub fn settle_ai_operation_once(
        &self,
        user_id: i64,
        chat_id: Option<i64>,
        operation_id: &str,
        actual_credit_units: i64,
        metadata: &Map<String, Value>,
    ) -> Result<AiSettlementResult, BillingError> {
        let actual_credit_units = actual_credit_units.max(0);
        self.run_transaction(|transaction| {
            let user_balance = Self::balance_for_update(transaction, BillingScope::User, user_id)?;
            let chat_balance = match chat_id {
                Some(chat_id) => {
                    Self::balance_for_update(transaction, BillingScope::Chat, chat_id)?
                }
                None => 0,
            };
            if Self::ai_operation_is_settled(transaction, user_id, operation_id)? {
                return Ok(AiSettlementResult {
                    applied: false,
                    source: None,
                    authorized_credit_units: None,
                    actual_credit_units: None,
                    refunded_credit_units: None,
                    debt_applied_credit_units: None,
                    user_balance: user_balance.into(),
                    chat_balance: chat_balance.into(),
                });
            }

            let hold = transaction.query_one(
                "SELECT COALESCE(SUM(-amount), 0), \
                    COUNT(DISTINCT metadata->>'source'), MIN(metadata->>'source'), \
                    COALESCE( \
                        (ARRAY_AGG(metadata ORDER BY id) FILTER ( \
                            WHERE event_type = 'ai_reserve' \
                        ))[1], '{}'::jsonb \
                    ), COALESCE(TO_JSONB(ARRAY_AGG(DISTINCT metadata->>'settlement_id') \
                        FILTER (WHERE event_type = 'ai_reserve' \
                            AND NULLIF(metadata->>'settlement_id', '') IS NOT NULL)), '[]'::jsonb) \
                 FROM credit_ledger WHERE user_id = $1 \
                   AND event_type IN ('ai_reserve', 'ai_refund') \
                   AND metadata->>'operation_id' = $2",
                &[&user_id, &operation_id],
            )?;
            let authorized = hold.get::<_, i64>(0).max(0);
            let payer_count = hold.get::<_, i64>(1);
            if payer_count > 1 {
                return Err(BillingError::MultiplePayers);
            }
            let payer = if hold.get::<_, Option<String>>(2).as_deref() == Some("chat") {
                BillingScope::Chat
            } else {
                BillingScope::User
            };
            if payer == BillingScope::Chat && chat_id.is_none() {
                return Err(BillingError::ChatIdRequired);
            }

            let adjustment = authorized
                .checked_sub(actual_credit_units)
                .ok_or(BillingError::BalanceOverflow)?;
            let adjustment_i32 =
                i32::try_from(adjustment).map_err(|_| BillingError::BalanceOverflow)?;
            let (updated_user_balance, updated_chat_balance) = match payer {
                BillingScope::Chat => {
                    let payer_chat_id = chat_id.ok_or(BillingError::ChatIdRequired)?;
                    let updated_chat_balance = chat_balance
                        .checked_add(adjustment_i32)
                        .ok_or(BillingError::BalanceOverflow)?;
                    Self::set_balance(
                        transaction,
                        BillingScope::Chat,
                        payer_chat_id,
                        updated_chat_balance,
                    )?;
                    (user_balance, updated_chat_balance)
                }
                BillingScope::User => {
                    let updated_user_balance = user_balance
                        .checked_add(adjustment_i32)
                        .ok_or(BillingError::BalanceOverflow)?;
                    Self::set_balance(
                        transaction,
                        BillingScope::User,
                        user_id,
                        updated_user_balance,
                    )?;
                    (updated_user_balance, chat_balance)
                }
            };
            let refunded = adjustment.max(0);
            let debt = if adjustment < 0 {
                adjustment
                    .checked_neg()
                    .ok_or(BillingError::BalanceOverflow)?
            } else {
                0
            };
            let mut merged_metadata = hold
                .get::<_, Value>(3)
                .as_object()
                .cloned()
                .unwrap_or_default();
            merged_metadata.extend(metadata.clone());
            let settlement_ids = hold.get::<_, Value>(4);
            if settlement_ids.as_array().is_some_and(|ids| !ids.is_empty()) {
                merged_metadata.insert("settlement_ids".to_owned(), settlement_ids);
            }
            let settlement_metadata = settlement_metadata(
                &merged_metadata,
                operation_id,
                payer,
                authorized,
                actual_credit_units,
                refunded,
                debt,
            );
            if adjustment != 0 {
                let event_type = if adjustment > 0 {
                    "ai_refund"
                } else {
                    "ai_settlement_debt"
                };
                transaction.execute(
                    "INSERT INTO credit_ledger \
                        (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
                     VALUES ($1, $2, $2, $3, $4, $5)",
                    &[
                        &event_type,
                        &user_id,
                        &chat_id,
                        &adjustment_i32,
                        &settlement_metadata,
                    ],
                )?;
            }
            transaction.execute(
                "INSERT INTO credit_ledger \
                    (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
                 VALUES ('ai_settlement_result', $1, $1, $2, 0, $3)",
                &[&user_id, &chat_id, &settlement_metadata],
            )?;
            Ok(AiSettlementResult {
                applied: true,
                source: Some(payer.as_str().to_owned()),
                authorized_credit_units: Some(authorized),
                actual_credit_units: Some(actual_credit_units),
                refunded_credit_units: Some(refunded),
                debt_applied_credit_units: Some(debt),
                user_balance: updated_user_balance.into(),
                chat_balance: updated_chat_balance.into(),
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn settle_legacy_ai_reservation_once(
        &self,
        user_id: i64,
        chat_id: Option<i64>,
        source: &str,
        reserved_credit_units: i64,
        actual_credit_units: i64,
        usage_tag: &str,
        metadata: &Map<String, Value>,
    ) -> Result<LegacySettlementResult, BillingError> {
        let payer = if source == "chat" {
            BillingScope::Chat
        } else {
            BillingScope::User
        };
        if payer == BillingScope::Chat && chat_id.is_none() {
            return Err(BillingError::LegacyChatIdRequired);
        }
        let reserved_credit_units = reserved_credit_units.max(0);
        let actual_credit_units = actual_credit_units.max(0);
        self.run_transaction(|transaction| {
            let user_balance = Self::balance_for_update(transaction, BillingScope::User, user_id)?;
            let chat_balance = match chat_id {
                Some(chat_id) => {
                    Self::balance_for_update(transaction, BillingScope::Chat, chat_id)?
                }
                None => 0,
            };
            let already_settled = transaction
                .query_opt(
                    "SELECT 1 FROM credit_ledger \
                     WHERE event_type = 'memory_compaction_settlement' \
                       AND metadata->>'usage_tag' = $1 LIMIT 1",
                    &[&usage_tag],
                )?
                .is_some();
            if already_settled {
                return Ok(LegacySettlementResult {
                    applied: false,
                    adjustment_credit_units: None,
                    user_balance: user_balance.into(),
                    chat_balance: chat_balance.into(),
                });
            }

            let adjustment = reserved_credit_units
                .checked_sub(actual_credit_units)
                .ok_or(BillingError::BalanceOverflow)?;
            let adjustment_i32 =
                i32::try_from(adjustment).map_err(|_| BillingError::BalanceOverflow)?;
            let (updated_user_balance, updated_chat_balance) = match payer {
                BillingScope::Chat => {
                    let payer_chat_id = chat_id.ok_or(BillingError::LegacyChatIdRequired)?;
                    let updated_chat_balance = chat_balance
                        .checked_add(adjustment_i32)
                        .ok_or(BillingError::BalanceOverflow)?;
                    Self::set_balance(
                        transaction,
                        BillingScope::Chat,
                        payer_chat_id,
                        updated_chat_balance,
                    )?;
                    (user_balance, updated_chat_balance)
                }
                BillingScope::User => {
                    let updated_user_balance = user_balance
                        .checked_add(adjustment_i32)
                        .ok_or(BillingError::BalanceOverflow)?;
                    Self::set_balance(
                        transaction,
                        BillingScope::User,
                        user_id,
                        updated_user_balance,
                    )?;
                    (updated_user_balance, chat_balance)
                }
            };
            let settlement_metadata = legacy_settlement_metadata(
                metadata,
                payer,
                usage_tag,
                reserved_credit_units,
                actual_credit_units,
                adjustment,
            );
            transaction.execute(
                "INSERT INTO credit_ledger \
                    (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
                 VALUES ('memory_compaction_settlement', $1, $1, $2, $3, $4)",
                &[&user_id, &chat_id, &adjustment_i32, &settlement_metadata],
            )?;
            Ok(LegacySettlementResult {
                applied: true,
                adjustment_credit_units: Some(adjustment),
                user_balance: updated_user_balance.into(),
                chat_balance: updated_chat_balance.into(),
            })
        })
    }

    pub fn record_ai_settlement_result(
        &self,
        user_id: i64,
        chat_id: Option<i64>,
        actor_user_id: i64,
        event_type: &str,
        metadata: &Map<String, Value>,
    ) -> Result<bool, BillingError> {
        self.run_transaction(|transaction| {
            Ok(transaction.execute(
                "INSERT INTO credit_ledger \
                    (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
                 VALUES ($1, $2, $3, $4, 0, $5) ON CONFLICT DO NOTHING",
                &[
                    &event_type,
                    &actor_user_id,
                    &user_id,
                    &chat_id,
                    &Value::Object(metadata.clone()),
                ],
            )? > 0)
        })
    }

    pub fn list_recent_ai_settlement_results(
        &self,
        limit: i64,
    ) -> Result<Vec<LedgerEntry>, BillingError> {
        let mut client = self.connect()?;
        client
            .query(
                "SELECT id, event_type, actor_user_id, user_id, chat_id, amount, \
                    metadata, created_at::text FROM credit_ledger \
                 WHERE event_type = 'ai_settlement_result' \
                 ORDER BY created_at DESC, id DESC LIMIT $1",
                &[&limit],
            )?
            .into_iter()
            .map(|row| {
                Ok(LedgerEntry {
                    id: row.try_get(0)?,
                    event_type: row.try_get(1)?,
                    actor_user_id: row.try_get(2)?,
                    user_id: row.try_get(3)?,
                    chat_id: row.try_get(4)?,
                    amount: row.try_get(5)?,
                    metadata: row.try_get(6)?,
                    created_at: row.try_get(7)?,
                })
            })
            .collect()
    }

    pub fn list_unsettled_ai_operations(
        &self,
        limit: i64,
    ) -> Result<Vec<UnsettledAiOperation>, BillingError> {
        let mut client = self.connect()?;
        client
            .query(
                "WITH pending AS ( \
                    SELECT ledger.metadata->>'operation_id' AS operation_id, \
                        MIN(ledger.user_id) AS user_id, MIN(ledger.chat_id) AS chat_id, \
                        GREATEST(0, COALESCE(SUM(-ledger.amount), 0)) AS authorized, \
                        MIN(ledger.metadata->>'source') AS source, \
                        MIN(ledger.created_at) AS created_at, \
                        MAX(ledger.created_at) AS hold_activity_at, \
                        (ARRAY_AGG(ledger.metadata ORDER BY ledger.id) \
                            FILTER (WHERE ledger.event_type = 'ai_reserve'))[1] \
                            AS reserve_metadata \
                    FROM credit_ledger AS ledger \
                    WHERE ledger.event_type IN ('ai_reserve', 'ai_refund') \
                      AND ledger.user_id IS NOT NULL \
                      AND ledger.metadata ? 'operation_id' \
                      AND NOT EXISTS ( \
                          SELECT 1 FROM credit_ledger AS background \
                          WHERE background.event_type = 'ai_reserve' \
                            AND background.metadata->>'operation_id' \
                                = ledger.metadata->>'operation_id' \
                            AND background.metadata->>'background' = 'true' \
                      ) \
                      AND NOT EXISTS ( \
                          SELECT 1 FROM credit_ledger AS settled WHERE ( \
                              settled.event_type = 'ai_settlement_result' \
                              AND settled.metadata->>'operation_id' \
                                  = ledger.metadata->>'operation_id' \
                          ) OR ( \
                              settled.event_type = 'memory_compaction_settlement' \
                              AND settled.user_id = ledger.user_id \
                              AND settled.metadata->>'usage_tag' \
                                  = ledger.metadata->>'usage_tag' \
                          ) \
                      ) \
                    GROUP BY ledger.metadata->>'operation_id' \
                ) \
                SELECT pending.operation_id, pending.user_id, pending.chat_id, \
                    pending.authorized, pending.source, pending.created_at::text, \
                    GREATEST( \
                        pending.hold_activity_at, \
                        COALESCE(( \
                            SELECT MAX(usage.created_at) FROM credit_ledger AS usage \
                            WHERE usage.event_type = 'ai_provider_usage' \
                              AND usage.metadata->>'operation_id' = pending.operation_id \
                        ), pending.hold_activity_at) \
                    )::text AS last_activity_at, pending.reserve_metadata, \
                    COALESCE(( \
                        SELECT jsonb_agg(jsonb_build_object( \
                            'segment_id', usage.metadata->>'segment_id', \
                            'segment', usage.metadata->'segment') ORDER BY usage.id) \
                        FROM credit_ledger AS usage \
                        WHERE usage.event_type = 'ai_provider_usage' \
                          AND usage.metadata->>'operation_id' = pending.operation_id \
                    ), '[]'::jsonb) AS segments \
                FROM pending \
                ORDER BY pending.created_at LIMIT $1",
                &[&limit],
            )?
            .into_iter()
            .map(|row| {
                let segments: Value = row.try_get(8)?;
                Ok(UnsettledAiOperation {
                    operation_id: row.try_get(0)?,
                    user_id: row.try_get(1)?,
                    chat_id: row.try_get(2)?,
                    authorized_credit_units: row.try_get(3)?,
                    source: if row.try_get::<_, Option<String>>(4)?.as_deref() == Some("chat") {
                        "chat".to_owned()
                    } else {
                        "user".to_owned()
                    },
                    created_at: row.try_get(5)?,
                    last_activity_at: row.try_get(6)?,
                    reserve_metadata: row
                        .try_get::<_, Option<Value>>(7)?
                        .unwrap_or_else(|| json!({})),
                    segments: segments.as_array().cloned().unwrap_or_default(),
                })
            })
            .collect()
    }

    pub fn purge_expired_ai_ledger_events(
        &self,
        retention_days: i64,
    ) -> Result<PurgeResult, BillingError> {
        self.run_transaction(|transaction| {
            let deleted_rows = transaction.execute(
                "DELETE FROM credit_ledger WHERE event_type IN ( \
                    'ai_reserve', 'ai_provider_usage', 'ai_refund', \
                    'ai_settlement_charge', 'ai_settlement_debt', \
                    'ai_settlement_result', 'ai_reconciliation_correction', \
                    'memory_compaction_settlement' \
                 ) AND created_at < NOW() - ($1::bigint * INTERVAL '1 day')",
                &[&retention_days],
            )?;
            Ok(PurgeResult {
                deleted_rows,
                retention_days,
            })
        })
    }

    pub fn list_user_ai_charge_rows(
        &self,
        user_id: i64,
        cursor_id: Option<i64>,
        direction: &str,
        group_limit: i64,
    ) -> Result<Vec<ChargeHistoryRow>, BillingError> {
        let mut client = self.connect()?;
        client
            .query(
                "WITH user_ledger AS ( \
                    SELECT * FROM credit_ledger WHERE user_id = $1 \
                      AND event_type IN ( \
                        'ai_settlement_result', 'memory_compaction_settlement', \
                        'ai_reserve', 'ai_refund', 'ai_settlement_charge', \
                        'ai_settlement_debt' \
                      ) \
                ), finalized_ids AS ( \
                    SELECT metadata->>'settlement_id' AS settlement_id \
                    FROM user_ledger \
                    WHERE event_type IN ( \
                        'ai_settlement_result', 'memory_compaction_settlement' \
                    ) AND metadata ? 'settlement_id' \
                    UNION \
                    SELECT settlement_id.value FROM user_ledger \
                    CROSS JOIN LATERAL jsonb_array_elements_text( \
                        CASE WHEN jsonb_typeof(metadata->'settlement_ids') = 'array' \
                            THEN metadata->'settlement_ids' ELSE '[]'::jsonb END \
                    ) AS settlement_id(value) \
                    WHERE event_type = 'ai_settlement_result' \
                ), finalized_operations AS ( \
                    SELECT settlement.id, settlement.event_type, \
                        settlement.actor_user_id, settlement.user_id, \
                        settlement.chat_id, settlement.amount, settlement.metadata, \
                        COALESCE(( \
                            SELECT MIN(reserve.created_at) FROM user_ledger AS reserve \
                            WHERE reserve.event_type = 'ai_reserve' AND CASE \
                                WHEN NULLIF(settlement.metadata->>'operation_id', '') IS NOT NULL \
                                    THEN reserve.metadata->>'operation_id' \
                                        = settlement.metadata->>'operation_id' \
                                WHEN NULLIF(settlement.metadata->>'settlement_id', '') IS NOT NULL \
                                    THEN reserve.metadata->>'settlement_id' \
                                        = settlement.metadata->>'settlement_id' \
                                ELSE NULLIF(settlement.metadata->>'usage_tag', '') IS NOT NULL \
                                    AND reserve.metadata->>'usage_tag' \
                                        = settlement.metadata->>'usage_tag' \
                            END \
                        ), settlement.created_at) AS created_at \
                    FROM user_ledger AS settlement \
                    WHERE event_type IN ( \
                        'ai_settlement_result', 'memory_compaction_settlement' \
                    ) AND NOT ( \
                        event_type = 'ai_settlement_result' AND EXISTS ( \
                            SELECT 1 FROM user_ledger AS legacy \
                            WHERE legacy.event_type = 'memory_compaction_settlement' \
                              AND legacy.metadata->>'usage_tag' \
                                  = settlement.metadata->>'usage_tag' \
                        ) \
                    ) \
                ), pending_reservations AS ( \
                    SELECT DISTINCT ON (metadata->>'settlement_id') \
                        id, event_type, actor_user_id, user_id, chat_id, amount, \
                        metadata, created_at \
                    FROM user_ledger AS reserve \
                    WHERE event_type = 'ai_reserve' \
                      AND metadata ? 'settlement_id' \
                      AND NOT EXISTS ( \
                        SELECT 1 FROM finalized_ids \
                        WHERE finalized_ids.settlement_id \
                            = reserve.metadata->>'settlement_id' \
                      ) \
                    ORDER BY metadata->>'settlement_id', id DESC \
                ), pending_operations AS ( \
                    SELECT reserve.id, reserve.event_type, reserve.actor_user_id, \
                        reserve.user_id, reserve.chat_id, \
                        -GREATEST(0, -totals.net_amount) AS amount, \
                        reserve.metadata || jsonb_build_object( \
                            'charged_credit_units_total', \
                                GREATEST(0, -totals.net_amount), \
                            'billing_pending', TRUE, \
                            'payer_scope', CASE \
                                WHEN totals.user_paid > 0 AND totals.chat_paid > 0 \
                                    THEN 'mixed' \
                                WHEN totals.chat_paid > 0 THEN 'chat' ELSE 'user' END, \
                            'payer_breakdown', jsonb_build_array( \
                                jsonb_build_object( \
                                    'scope', 'user', 'credit_units', totals.user_paid), \
                                jsonb_build_object( \
                                    'scope', 'chat', 'credit_units', totals.chat_paid) \
                            ) \
                        ) AS metadata, reserve.created_at \
                    FROM pending_reservations AS reserve \
                    CROSS JOIN LATERAL ( \
                        SELECT COALESCE(SUM(mutation.amount), 0) AS net_amount, \
                            GREATEST(0, COALESCE(SUM(-mutation.amount) FILTER ( \
                                WHERE mutation.metadata->>'source' = 'user'), 0)) \
                                AS user_paid, \
                            GREATEST(0, COALESCE(SUM(-mutation.amount) FILTER ( \
                                WHERE mutation.metadata->>'source' = 'chat'), 0)) \
                                AS chat_paid \
                        FROM user_ledger AS mutation \
                        WHERE mutation.event_type IN ( \
                            'ai_reserve', 'ai_refund', 'ai_settlement_charge', \
                            'ai_settlement_debt' \
                        ) AND mutation.metadata->>'settlement_id' \
                            = reserve.metadata->>'settlement_id' \
                    ) AS totals \
                ), operations AS ( \
                    SELECT * FROM finalized_operations \
                    UNION ALL SELECT * FROM pending_operations \
                ), billable_operations AS ( \
                    SELECT * FROM operations WHERE COALESCE( \
                        (metadata->>'charged_credit_units_total')::bigint, \
                        (metadata->>'actual_credit_units')::bigint, \
                        (metadata->>'settled_credit_units')::bigint, \
                        GREATEST(0, -amount::bigint), 0 \
                    ) > 0 \
                ), grouped_operations AS ( \
                    SELECT billable_operations.*, CONCAT( \
                        COALESCE( \
                            NULLIF(metadata->>'origin_chat_id', ''), \
                            NULLIF(SPLIT_PART(metadata->>'settlement_id', ':', 2), ''), \
                            chat_id::text, '' \
                        ), ':', COALESCE( \
                            NULLIF(metadata->>'message_id', ''), 'ledger:' || id::text \
                        ) \
                    ) AS group_key FROM billable_operations \
                ), charge_groups AS ( \
                    SELECT group_key, MIN(id) AS group_cursor, \
                        MIN(created_at) AS group_created_at \
                    FROM grouped_operations GROUP BY group_key \
                ), page_groups AS ( \
                    SELECT group_key, group_cursor, group_created_at \
                    FROM charge_groups \
                    WHERE $2::bigint IS NULL \
                       OR ($3::text = 'older' AND group_cursor < $2) \
                       OR ($3::text = 'newer' AND group_cursor > $2) \
                    ORDER BY \
                        CASE WHEN $3::text = 'newer' THEN group_cursor END ASC, \
                        CASE WHEN $3::text = 'older' THEN group_cursor END DESC \
                    LIMIT $4 \
                ) \
                SELECT operation.id, operation.event_type, \
                    operation.actor_user_id, operation.user_id, operation.chat_id, \
                    operation.amount, operation.metadata, operation.created_at::text, \
                    page.group_key, page.group_cursor, page.group_created_at::text \
                FROM page_groups AS page \
                JOIN grouped_operations AS operation USING (group_key) \
                ORDER BY \
                    CASE WHEN $3::text = 'newer' THEN page.group_cursor END ASC, \
                    CASE WHEN $3::text = 'older' THEN page.group_cursor END DESC, \
                    operation.id DESC",
                &[&user_id, &cursor_id, &direction, &group_limit],
            )?
            .into_iter()
            .map(|row| {
                Ok(ChargeHistoryRow {
                    id: row.try_get(0)?,
                    event_type: row.try_get(1)?,
                    actor_user_id: row.try_get(2)?,
                    user_id: row.try_get(3)?,
                    chat_id: row.try_get(4)?,
                    amount: row.try_get(5)?,
                    metadata: row.try_get(6)?,
                    created_at: row.try_get(7)?,
                    group_key: row.try_get(8)?,
                    group_cursor: row.try_get(9)?,
                    group_created_at: row.try_get(10)?,
                })
            })
            .collect()
    }

    fn ai_operation_is_settled(
        transaction: &mut Transaction<'_>,
        user_id: i64,
        operation_id: &str,
    ) -> Result<bool, BillingError> {
        if operation_id.is_empty() {
            return Ok(false);
        }
        Ok(transaction
            .query_opt(
                "SELECT 1 FROM credit_ledger \
                 WHERE user_id = $1 AND event_type = 'ai_settlement_result' \
                   AND metadata->>'operation_id' = $2 LIMIT 1",
                &[&user_id, &operation_id],
            )?
            .is_some())
    }

    fn rejected_ai_charge(reason: &str, user_balance: i32, chat_balance: i32) -> AiChargeResult {
        AiChargeResult {
            ok: false,
            applied: false,
            reason: Some(reason.to_owned()),
            source: None,
            amount: 0,
            user_balance: user_balance.into(),
            chat_balance: chat_balance.into(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_ai_charge_ledger(
        transaction: &mut Transaction<'_>,
        event_type: &str,
        user_id: i64,
        chat_id: Option<i64>,
        amount: i32,
        source: &str,
        metadata: &Map<String, Value>,
    ) -> Result<(), BillingError> {
        let metadata = billing_metadata(source, metadata);
        transaction.execute(
            "INSERT INTO credit_ledger \
                (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
             VALUES ($1, $2, $2, $3, $4, $5)",
            &[&event_type, &user_id, &chat_id, &amount, &metadata],
        )?;
        Ok(())
    }

    fn run_transaction<T, F>(&self, operation: F) -> Result<T, BillingError>
    where
        F: for<'transaction> Fn(&mut Transaction<'transaction>) -> Result<T, BillingError>,
    {
        for attempt in 0..CREDIT_TRANSACTION_MAX_ATTEMPTS {
            let result = (|| {
                let mut client = self.connect()?;
                let mut transaction = client.transaction()?;
                let result = operation(&mut transaction)?;
                transaction.commit()?;
                Ok(result)
            })();
            match result {
                Ok(result) => return Ok(result),
                Err(error)
                    if attempt + 1 < CREDIT_TRANSACTION_MAX_ATTEMPTS
                        && is_retryable_transaction_error(&error) =>
                {
                    continue;
                }
                Err(error) => return Err(error),
            }
        }
        Err(BillingError::TransactionRetriesExhausted)
    }

    fn balance_for_update(
        transaction: &mut Transaction<'_>,
        scope: BillingScope,
        scope_id: i64,
    ) -> Result<i32, BillingError> {
        transaction.execute(
            "INSERT INTO credit_accounts (scope_type, scope_id, balance) VALUES ($1, $2, 0) \
             ON CONFLICT (scope_type, scope_id) DO NOTHING",
            &[&scope.as_str(), &scope_id],
        )?;
        Ok(transaction
            .query_one(
                "SELECT balance FROM credit_accounts \
                 WHERE scope_type = $1 AND scope_id = $2 FOR UPDATE",
                &[&scope.as_str(), &scope_id],
            )?
            .get(0))
    }

    fn set_balance(
        transaction: &mut Transaction<'_>,
        scope: BillingScope,
        scope_id: i64,
        balance: i32,
    ) -> Result<(), BillingError> {
        transaction.execute(
            "UPDATE credit_accounts SET balance = $1, updated_at = NOW() \
             WHERE scope_type = $2 AND scope_id = $3",
            &[&balance, &scope.as_str(), &scope_id],
        )?;
        Ok(())
    }

    fn apply_onboarding_grant(
        transaction: &mut Transaction<'_>,
        user_id: i64,
        credits: i32,
    ) -> Result<OnboardingGrantResult, BillingError> {
        transaction.query_one(
            "SELECT pg_advisory_xact_lock($1)",
            &[&ONBOARDING_GRANTS_ADVISORY_LOCK_KEY],
        )?;
        let mut balance = Self::balance_for_update(transaction, BillingScope::User, user_id)?;
        if transaction
            .query_opt(
                "SELECT 1 FROM onboarding_grants WHERE user_id = $1",
                &[&user_id],
            )?
            .is_some()
        {
            return Ok(OnboardingGrantResult {
                granted: false,
                balance: balance.into(),
            });
        }

        let counts = transaction.query_one(
            "SELECT \
                COUNT(*) FILTER (WHERE granted_at >= NOW() - INTERVAL '1 hour'), \
                COUNT(*) FILTER (WHERE granted_at >= NOW() - INTERVAL '1 day') \
             FROM onboarding_grants",
            &[],
        )?;
        let hourly_count = counts.get::<_, i64>(0);
        let daily_count = counts.get::<_, i64>(1);
        if hourly_count >= ONBOARDING_MAX_GRANTS_PER_HOUR
            || daily_count >= ONBOARDING_MAX_GRANTS_PER_DAY
        {
            let metadata = json!({
                "credits": credits,
                "hourly_count": hourly_count,
                "daily_count": daily_count,
                "hourly_limit": ONBOARDING_MAX_GRANTS_PER_HOUR,
                "daily_limit": ONBOARDING_MAX_GRANTS_PER_DAY,
            });
            transaction.execute(
                "INSERT INTO credit_ledger \
                    (event_type, actor_user_id, user_id, amount, metadata) \
                 VALUES ('onboarding_denied_overflow', $1, $1, 0, $2)",
                &[&user_id, &metadata],
            )?;
            return Ok(OnboardingGrantResult {
                granted: false,
                balance: balance.into(),
            });
        }

        let granted = transaction
            .query_opt(
                "INSERT INTO onboarding_grants (user_id, credits) VALUES ($1, $2) \
                 ON CONFLICT (user_id) DO NOTHING RETURNING user_id",
                &[&user_id, &credits],
            )?
            .is_some();
        if granted {
            balance = balance
                .checked_add(credits)
                .ok_or(BillingError::BalanceOverflow)?;
            Self::set_balance(transaction, BillingScope::User, user_id, balance)?;
            let metadata = json!({"credits": credits});
            transaction.execute(
                "INSERT INTO credit_ledger \
                    (event_type, actor_user_id, user_id, amount, metadata) \
                 VALUES ('onboarding_grant', $1, $1, $2, $3)",
                &[&user_id, &credits, &metadata],
            )?;
        }
        Ok(OnboardingGrantResult {
            granted,
            balance: balance.into(),
        })
    }

    fn connect(&self) -> Result<Client, BillingError> {
        Ok(Client::connect(
            &self.database_url,
            postgres_tls_connector(&self.database_url)?,
        )?)
    }
}

fn is_retryable_transaction_error(error: &BillingError) -> bool {
    let BillingError::Postgres(error) = error else {
        return false;
    };
    error.code().is_some_and(|code| {
        *code == SqlState::T_R_SERIALIZATION_FAILURE || *code == SqlState::T_R_DEADLOCK_DETECTED
    })
}

fn billing_metadata(source: &str, metadata: &Map<String, Value>) -> Value {
    let mut merged = metadata.clone();
    merged
        .entry("source".to_owned())
        .or_insert_with(|| Value::String(source.to_owned()));
    Value::Object(merged)
}

fn ai_mutation_metadata(
    metadata: &Map<String, Value>,
    idempotency_key: Option<&str>,
    operation_id: &str,
) -> Map<String, Value> {
    let mut merged = metadata.clone();
    if let Some(idempotency_key) = idempotency_key.filter(|key| !key.is_empty()) {
        merged.insert(
            "idempotency_key".to_owned(),
            Value::String(idempotency_key.to_owned()),
        );
        merged
            .entry("settlement_id".to_owned())
            .or_insert_with(|| Value::String(idempotency_key.to_owned()));
    }
    if !operation_id.is_empty() {
        merged.insert(
            "operation_id".to_owned(),
            Value::String(operation_id.to_owned()),
        );
    }
    merged
}

#[allow(clippy::too_many_arguments)]
fn settlement_metadata(
    metadata: &Map<String, Value>,
    operation_id: &str,
    payer: BillingScope,
    authorized: i64,
    actual: i64,
    refunded: i64,
    debt: i64,
) -> Value {
    let mut merged = metadata.clone();
    merged.insert(
        "operation_id".to_owned(),
        Value::String(operation_id.to_owned()),
    );
    merged.insert(
        "source".to_owned(),
        Value::String(payer.as_str().to_owned()),
    );
    merged.insert(
        "payer_scope".to_owned(),
        Value::String(payer.as_str().to_owned()),
    );
    for (key, value) in [
        ("reserved_credit_units_total", authorized),
        ("settled_credit_units", actual),
        ("refunded_credit_units", refunded),
        ("debt_applied_credit_units", debt),
        ("charged_credit_units_total", actual),
    ] {
        merged.insert(key.to_owned(), Value::from(value));
    }
    Value::Object(merged)
}

fn legacy_settlement_metadata(
    metadata: &Map<String, Value>,
    payer: BillingScope,
    usage_tag: &str,
    reserved: i64,
    actual: i64,
    adjustment: i64,
) -> Value {
    let mut merged = Map::from_iter([
        (
            "source".to_owned(),
            Value::String(payer.as_str().to_owned()),
        ),
        ("usage_tag".to_owned(), Value::String(usage_tag.to_owned())),
        ("reserved_credit_units".to_owned(), Value::from(reserved)),
        ("actual_credit_units".to_owned(), Value::from(actual)),
        (
            "adjustment_credit_units".to_owned(),
            Value::from(adjustment),
        ),
    ]);
    merged.extend(metadata.clone());
    Value::Object(merged)
}

#[cfg(test)]
mod tests {
    use native_tls::TlsConnector;
    use postgres::Client;
    use postgres_native_tls::MakeTlsConnector;
    use serde_json::{Value, json};

    use super::{
        AiChargeResult, AiRefundResult, AiSettlementResult, BalancePairResult, BillingError,
        BillingRepository, BillingScope, ChatAiChargeResult, LegacySettlementResult,
        OnboardingGrantResult, PurgeResult, StarPaymentResult, TransferResult,
        ai_mutation_metadata,
    };
    use crate::billing_schema::{BillingSchemaRepository, BillingSchemaResult};

    #[test]
    fn validates_the_persistent_scope_contract_before_connecting() {
        assert_eq!(BillingScope::parse("user").ok(), Some(BillingScope::User));
        assert_eq!(BillingScope::parse("chat").ok(), Some(BillingScope::Chat));
        let repository = BillingRepository::new("postgresql://invalid.invalid/db");
        assert!(matches!(
            repository.get_balance("group", 1),
            Err(BillingError::InvalidScope(scope)) if scope == "group"
        ));
        assert!(matches!(
            repository.get_or_create_balance("group", 1),
            Err(BillingError::InvalidScope(scope)) if scope == "group"
        ));
    }

    #[test]
    fn ai_mutations_persist_one_key_for_replay_and_settlement_identity() {
        let metadata = serde_json::Map::from_iter([("trace_id".to_owned(), json!("trace"))]);
        let enriched = ai_mutation_metadata(&metadata, Some("reservation-1"), "operation-1");
        assert_eq!(enriched["idempotency_key"], "reservation-1");
        assert_eq!(enriched["settlement_id"], "reservation-1");
        assert_eq!(enriched["operation_id"], "operation-1");
        assert_eq!(enriched["trace_id"], "trace");

        let explicit_settlement = serde_json::Map::from_iter([(
            "settlement_id".to_owned(),
            json!("original-reservation"),
        )]);
        let refund = ai_mutation_metadata(&explicit_settlement, Some("refund-1"), "operation-1");
        assert_eq!(refund["idempotency_key"], "refund-1");
        assert_eq!(refund["settlement_id"], "original-reservation");
    }

    #[test]
    fn reads_existing_and_missing_balances_when_test_postgres_is_available()
    -> Result<(), Box<dyn std::error::Error>> {
        let Ok(database_url) = std::env::var("TEST_POSTGRES_URL") else {
            return Ok(());
        };
        let connector = TlsConnector::builder().build()?;
        let mut client = Client::connect(&database_url, MakeTlsConnector::new(connector))?;
        BillingSchemaRepository::new(&database_url).ensure_schema()?;
        client.batch_execute(
            "DELETE FROM star_payments \
                WHERE user_id BETWEEN 7000000000001 AND 7000000000999 \
                   OR user_id BETWEEN 8100000000000 AND 8100999999999; \
             DELETE FROM credit_ledger \
                WHERE user_id BETWEEN 7000000000001 AND 7000000000999 \
                   OR chat_id BETWEEN 7000000000001 AND 7000000000999 \
                   OR user_id BETWEEN 8100000000000 AND 8100999999999 \
                   OR chat_id BETWEEN -8200999999999 AND -8200000000000; \
             DELETE FROM onboarding_grants \
                WHERE user_id BETWEEN 7000000000001 AND 7000000000999 \
                   OR user_id BETWEEN 8100000000000 AND 8100999999999; \
             DELETE FROM credit_accounts \
                WHERE scope_id BETWEEN 7000000000001 AND 7000000000999 \
                   OR scope_id BETWEEN 8100000000000 AND 8100999999999 \
                   OR scope_id BETWEEN -8200999999999 AND -8200000000000; \
             DELETE FROM star_payments \
                WHERE user_id IN (7000000000043, 7000000000044); \
             DELETE FROM credit_ledger \
                WHERE user_id IN (7000000000043, 7000000000044) \
                   OR chat_id IN (7000000000043, 7000000000044); \
             DELETE FROM onboarding_grants \
                WHERE user_id IN (7000000000043, 7000000000044); \
             DELETE FROM credit_accounts \
                WHERE scope_id IN (7000000000043, 7000000000044); \
             DELETE FROM credit_schema_migrations WHERE name IN ( \
                'credit_amounts_scaled_to_tenths_v1', \
                'credit_amounts_scaled_to_hundredths_v2', \
                'repair_duplicate_compaction_refunds_v1' \
             ); \
             INSERT INTO credit_accounts (scope_type, scope_id, balance) VALUES \
                ('user', 7000000000043, 3), \
                ('user', 7000000000044, 100); \
             INSERT INTO onboarding_grants (user_id, credits) \
                VALUES (7000000000043, 3); \
             INSERT INTO star_payments (telegram_payment_charge_id, user_id, \
                pack_id, xtr_amount, credits_awarded, payload) \
                VALUES ('synthetic-schema-scale', 7000000000043, 'probe', 1, 100, NULL); \
             INSERT INTO credit_ledger (event_type, actor_user_id, user_id, amount, metadata) \
                VALUES ('synthetic_schema_scale', 7000000000043, 7000000000043, -3, \
                    '{\"reserved_credit_units\":3,\"note\":\"keep\"}'); \
             INSERT INTO credit_ledger (event_type, actor_user_id, user_id, amount, metadata) \
                VALUES \
                ('memory_compaction_settlement', 7000000000044, 7000000000044, 0, \
                    '{\"usage_tag\":\"memory_compaction:synthetic:schema\"}'), \
                ('ai_settlement_result', 7000000000044, 7000000000044, 0, \
                    '{\"operation_id\":\"synthetic-schema-repair\",\"settled_credit_units\":0}'), \
                ('ai_refund', 7000000000044, 7000000000044, 5, \
                    '{\"source\":\"user\",\"operation_id\":\"synthetic-schema-repair\",\
                      \"usage_tag\":\"memory_compaction:synthetic:schema\",\
                      \"reason\":\"unused_stale_reservation\"}');",
        )?;

        let first_schema_url = database_url.clone();
        let second_schema_url = database_url.clone();
        let first_schema = std::thread::spawn(move || {
            BillingSchemaRepository::new(&first_schema_url).ensure_schema()
        });
        let second_schema = std::thread::spawn(move || {
            BillingSchemaRepository::new(&second_schema_url).ensure_schema()
        });
        let first_schema = first_schema
            .join()
            .map_err(|_| std::io::Error::other("first schema migration thread panicked"))??;
        let second_schema = second_schema
            .join()
            .map_err(|_| std::io::Error::other("second schema migration thread panicked"))??;
        let schema_results = [first_schema, second_schema];
        assert_eq!(
            schema_results
                .iter()
                .filter(|result| result.migrated_to_tenths)
                .count(),
            1
        );
        assert_eq!(
            schema_results
                .iter()
                .filter(|result| result.migrated_to_hundredths)
                .count(),
            1
        );
        assert_eq!(
            schema_results
                .iter()
                .map(|result| result.repaired_compaction_refunds)
                .sum::<u64>(),
            1
        );
        let schema_evidence = client.query_one(
            "SELECT \
                (SELECT balance FROM credit_accounts \
                    WHERE scope_type = 'user' AND scope_id = 7000000000043), \
                (SELECT credits FROM onboarding_grants \
                    WHERE user_id = 7000000000043), \
                (SELECT credits_awarded FROM star_payments \
                    WHERE telegram_payment_charge_id = 'synthetic-schema-scale'), \
                (SELECT amount FROM credit_ledger \
                    WHERE user_id = 7000000000043 \
                      AND event_type = 'synthetic_schema_scale'), \
                (SELECT metadata FROM credit_ledger \
                    WHERE user_id = 7000000000043 \
                      AND event_type = 'synthetic_schema_scale'), \
                (SELECT balance FROM credit_accounts \
                    WHERE scope_type = 'user' AND scope_id = 7000000000044), \
                (SELECT COUNT(*) FROM credit_schema_migrations), \
                (SELECT COUNT(*) FROM pg_indexes WHERE indexname IN ( \
                    'idx_credit_ledger_compaction_usage_tag', \
                    'idx_credit_ledger_user_ai_settlements', \
                    'idx_credit_ledger_unique_ai_settlement', \
                    'idx_credit_ledger_settlement_id', \
                    'idx_credit_ledger_unique_ai_provider_segment', \
                    'idx_credit_ledger_user_charge_history', \
                    'idx_credit_ledger_user_charge_operations', \
                    'idx_credit_ledger_user_settlement_lookup' \
                ))",
            &[],
        )?;
        assert_eq!(schema_evidence.get::<_, i32>(0), 300);
        assert_eq!(schema_evidence.get::<_, i32>(1), 300);
        assert_eq!(schema_evidence.get::<_, i32>(2), 10_000);
        assert_eq!(schema_evidence.get::<_, i32>(3), -300);
        assert_eq!(
            schema_evidence.get::<_, Value>(4),
            json!({"reserved_credit_units": 30, "note": "keep", "credit_scale": 100})
        );
        assert_eq!(schema_evidence.get::<_, i32>(5), 9_500);
        assert_eq!(schema_evidence.get::<_, i64>(6), 3);
        assert_eq!(schema_evidence.get::<_, i64>(7), 8);
        assert_eq!(
            BillingSchemaRepository::new(&database_url).ensure_schema()?,
            BillingSchemaResult {
                migrated_to_tenths: false,
                migrated_to_hundredths: false,
                repaired_compaction_refunds: 0,
            }
        );
        client.execute(
            "INSERT INTO credit_accounts (scope_type, scope_id, balance) \
             VALUES ('user', $1, 1234) \
             ON CONFLICT (scope_type, scope_id) DO UPDATE SET balance = EXCLUDED.balance",
            &[&7_000_000_000_001_i64],
        )?;
        drop(client);

        let repository = BillingRepository::new(&database_url);
        assert_eq!(repository.get_balance("user", 7_000_000_000_001)?, 1234);
        assert_eq!(repository.get_balance("chat", 7_000_000_000_002)?, 0);
        assert_eq!(
            repository.get_or_create_balance("chat", 7_000_000_000_002)?,
            0
        );

        let connector = TlsConnector::builder().build()?;
        let mut client = Client::connect(&database_url, MakeTlsConnector::new(connector))?;
        assert_eq!(
            client
                .query_one(
                    "SELECT COUNT(*) FROM credit_accounts WHERE scope_type = 'chat' AND scope_id = $1",
                    &[&7_000_000_000_002_i64],
                )?
                .get::<_, i64>(0),
            1
        );
        assert_eq!(
            repository.grant_onboarding_if_needed(7_000_000_000_003, 300)?,
            OnboardingGrantResult {
                granted: true,
                balance: 300,
            }
        );
        assert_eq!(
            repository.grant_onboarding_if_needed(7_000_000_000_003, 300)?,
            OnboardingGrantResult {
                granted: false,
                balance: 300,
            }
        );
        let first_database_url = database_url.clone();
        let second_database_url = database_url.clone();
        let first = std::thread::spawn(move || {
            BillingRepository::new(&first_database_url)
                .grant_onboarding_if_needed(7_000_000_000_004, 300)
        });
        let second = std::thread::spawn(move || {
            BillingRepository::new(&second_database_url)
                .grant_onboarding_if_needed(7_000_000_000_004, 300)
        });
        let concurrent_results = [
            first
                .join()
                .map_err(|_| std::io::Error::other("first onboarding thread panicked"))??,
            second
                .join()
                .map_err(|_| std::io::Error::other("second onboarding thread panicked"))??,
        ];
        assert_eq!(
            concurrent_results
                .iter()
                .filter(|result| result.granted)
                .count(),
            1
        );
        assert!(
            concurrent_results
                .iter()
                .all(|result| result.balance == 300)
        );
        client.batch_execute(
            "INSERT INTO onboarding_grants (user_id, credits) VALUES \
                (7000000000004, 1), (7000000000005, 1), (7000000000006, 1) \
             ON CONFLICT (user_id) DO NOTHING;",
        )?;
        assert_eq!(
            repository.grant_onboarding_if_needed(7_000_000_000_007, 300)?,
            OnboardingGrantResult {
                granted: false,
                balance: 0,
            }
        );
        let event_counts = client.query_one(
            "SELECT \
                COUNT(*) FILTER (WHERE event_type = 'onboarding_grant'), \
                COUNT(*) FILTER (WHERE event_type = 'onboarding_denied_overflow') \
             FROM credit_ledger WHERE user_id >= $1 AND user_id <= $2",
            &[&7_000_000_000_001_i64, &7_000_000_000_007_i64],
        )?;
        assert_eq!(event_counts.get::<_, i64>(0), 2);
        assert_eq!(event_counts.get::<_, i64>(1), 1);

        assert_eq!(
            repository.record_star_payment(
                "synthetic-charge-1",
                7_000_000_000_008,
                "small",
                100,
                500,
                Some("synthetic-payload"),
            )?,
            StarPaymentResult {
                inserted: true,
                user_balance: 500,
            }
        );
        assert_eq!(
            repository.record_star_payment(
                "synthetic-charge-1",
                7_000_000_000_008,
                "small",
                100,
                500,
                Some("synthetic-payload"),
            )?,
            StarPaymentResult {
                inserted: false,
                user_balance: 500,
            }
        );
        let first_database_url = database_url.clone();
        let second_database_url = database_url.clone();
        let first = std::thread::spawn(move || {
            BillingRepository::new(&first_database_url).record_star_payment(
                "synthetic-charge-2",
                7_000_000_000_008,
                "small",
                100,
                500,
                None,
            )
        });
        let second = std::thread::spawn(move || {
            BillingRepository::new(&second_database_url).record_star_payment(
                "synthetic-charge-2",
                7_000_000_000_008,
                "small",
                100,
                500,
                None,
            )
        });
        let concurrent_payment_results = [
            first
                .join()
                .map_err(|_| std::io::Error::other("first Stars payment thread panicked"))??,
            second
                .join()
                .map_err(|_| std::io::Error::other("second Stars payment thread panicked"))??,
        ];
        assert_eq!(
            concurrent_payment_results
                .iter()
                .filter(|result| result.inserted)
                .count(),
            1
        );
        assert!(
            concurrent_payment_results
                .iter()
                .all(|result| result.user_balance == 1000)
        );
        let topup_evidence = client.query_one(
            "SELECT COUNT(*), \
                COUNT(*) FILTER (\
                    WHERE metadata->>'pack_id' = 'small' \
                      AND metadata->>'xtr_amount' = '100' \
                      AND metadata ? 'charge_id'\
                ) \
             FROM credit_ledger \
             WHERE user_id = $1 AND event_type = 'topup'",
            &[&7_000_000_000_008_i64],
        )?;
        assert_eq!(topup_evidence.get::<_, i64>(0), 2);
        assert_eq!(topup_evidence.get::<_, i64>(1), 2);

        assert_eq!(
            repository.mint_user_credits(7_000_000_000_009, 500, Some(99))?,
            500
        );
        assert_eq!(
            repository.transfer_user_to_chat(7_000_000_000_009, 7_000_000_000_010, 300,)?,
            TransferResult {
                transferred: true,
                user_balance: 200,
                chat_balance: 300,
            }
        );
        assert_eq!(
            repository.transfer_user_to_chat(7_000_000_000_009, 7_000_000_000_010, 500,)?,
            TransferResult {
                transferred: false,
                user_balance: 200,
                chat_balance: 300,
            }
        );
        let manual_evidence = client.query_one(
            "SELECT \
                COUNT(*) FILTER (WHERE event_type = 'printcredits'), \
                COUNT(*) FILTER (WHERE event_type = 'transfer_user_to_chat') \
             FROM credit_ledger WHERE user_id = $1",
            &[&7_000_000_000_009_i64],
        )?;
        assert_eq!(manual_evidence.get::<_, i64>(0), 1);
        assert_eq!(manual_evidence.get::<_, i64>(1), 2);

        assert_eq!(
            repository.mint_user_credits(7_000_000_000_011, 500, None)?,
            500
        );
        let first_database_url = database_url.clone();
        let second_database_url = database_url.clone();
        let first = std::thread::spawn(move || {
            BillingRepository::new(&first_database_url).transfer_user_to_chat(
                7_000_000_000_011,
                7_000_000_000_012,
                300,
            )
        });
        let second = std::thread::spawn(move || {
            BillingRepository::new(&second_database_url).transfer_user_to_chat(
                7_000_000_000_011,
                7_000_000_000_012,
                300,
            )
        });
        let concurrent_transfer_results = [
            first
                .join()
                .map_err(|_| std::io::Error::other("first transfer thread panicked"))??,
            second
                .join()
                .map_err(|_| std::io::Error::other("second transfer thread panicked"))??,
        ];
        assert_eq!(
            concurrent_transfer_results
                .iter()
                .filter(|result| result.transferred)
                .count(),
            1
        );
        assert!(concurrent_transfer_results.iter().any(|result| {
            result.transferred && result.user_balance == 200 && result.chat_balance == 300
        }));
        assert!(concurrent_transfer_results.iter().any(|result| {
            !result.transferred && result.user_balance == 200 && result.chat_balance == 300
        }));

        let first_database_url = database_url.clone();
        let second_database_url = database_url.clone();
        let first = std::thread::spawn(move || {
            BillingRepository::new(&first_database_url).mint_user_credits(
                7_000_000_000_013,
                500,
                Some(99),
            )
        });
        let second = std::thread::spawn(move || {
            BillingRepository::new(&second_database_url).mint_user_credits(
                7_000_000_000_013,
                500,
                Some(99),
            )
        });
        let mut concurrent_mint_balances = [
            first
                .join()
                .map_err(|_| std::io::Error::other("first mint thread panicked"))??,
            second
                .join()
                .map_err(|_| std::io::Error::other("second mint thread panicked"))??,
        ];
        concurrent_mint_balances.sort_unstable();
        assert_eq!(concurrent_mint_balances, [500, 1000]);

        client.execute(
            "INSERT INTO credit_accounts (scope_type, scope_id, balance) \
             VALUES ('chat', $1, 500) \
             ON CONFLICT (scope_type, scope_id) DO UPDATE SET balance = EXCLUDED.balance",
            &[&7_000_000_000_014_i64],
        )?;
        let custom_metadata = json!({
            "operation_id": "synthetic-chat-automation",
            "source": "automation"
        });
        let custom_metadata = custom_metadata
            .as_object()
            .ok_or_else(|| std::io::Error::other("synthetic metadata must be an object"))?;
        assert_eq!(
            repository.charge_chat_ai_credits(
                7_000_000_000_014,
                200,
                "ai_reserve",
                custom_metadata,
            )?,
            ChatAiChargeResult {
                charged: true,
                chat_balance: 300,
            }
        );
        assert_eq!(
            repository.charge_chat_ai_credits(
                7_000_000_000_014,
                400,
                "ai_reserve",
                custom_metadata,
            )?,
            ChatAiChargeResult {
                charged: false,
                chat_balance: 300,
            }
        );
        assert_eq!(
            repository.refund_chat_ai_credits(
                7_000_000_000_014,
                100,
                "ai_refund",
                custom_metadata,
            )?,
            400
        );
        assert_eq!(
            repository.apply_chat_ai_debt(
                7_000_000_000_014,
                650,
                "ai_settlement_debt",
                custom_metadata,
            )?,
            -250
        );
        let chat_ai_evidence = client.query_one(
            "SELECT COUNT(*), \
                COUNT(*) FILTER (WHERE metadata->>'source' = 'automation'), \
                COALESCE(SUM(amount), 0) \
             FROM credit_ledger WHERE chat_id = $1",
            &[&7_000_000_000_014_i64],
        )?;
        assert_eq!(chat_ai_evidence.get::<_, i64>(0), 3);
        assert_eq!(chat_ai_evidence.get::<_, i64>(1), 3);
        assert_eq!(chat_ai_evidence.get::<_, i64>(2), -750);

        client.execute(
            "INSERT INTO credit_accounts (scope_type, scope_id, balance) \
             VALUES ('chat', $1, 500) \
             ON CONFLICT (scope_type, scope_id) DO UPDATE SET balance = EXCLUDED.balance",
            &[&7_000_000_000_015_i64],
        )?;
        let first_database_url = database_url.clone();
        let second_database_url = database_url.clone();
        let first = std::thread::spawn(move || {
            BillingRepository::new(&first_database_url).charge_chat_ai_credits(
                7_000_000_000_015,
                300,
                "ai_reserve",
                &serde_json::Map::new(),
            )
        });
        let second = std::thread::spawn(move || {
            BillingRepository::new(&second_database_url).charge_chat_ai_credits(
                7_000_000_000_015,
                300,
                "ai_reserve",
                &serde_json::Map::new(),
            )
        });
        let concurrent_chat_charge_results = [
            first
                .join()
                .map_err(|_| std::io::Error::other("first chat charge thread panicked"))??,
            second
                .join()
                .map_err(|_| std::io::Error::other("second chat charge thread panicked"))??,
        ];
        assert_eq!(
            concurrent_chat_charge_results
                .iter()
                .filter(|result| result.charged)
                .count(),
            1
        );
        assert!(
            concurrent_chat_charge_results
                .iter()
                .all(|result| result.chat_balance == 200)
        );

        client.batch_execute(
            "INSERT INTO credit_accounts (scope_type, scope_id, balance) VALUES \
                ('user', 7000000000016, 500), \
                ('chat', 7000000000017, 700) \
             ON CONFLICT (scope_type, scope_id) DO UPDATE SET balance = EXCLUDED.balance;",
        )?;
        assert_eq!(
            repository.apply_ai_debt(
                7_000_000_000_016,
                Some(7_000_000_000_017),
                900,
                "chat",
                "ai_settlement_debt",
                &serde_json::Map::new(),
            )?,
            BalancePairResult {
                user_balance: 500,
                chat_balance: -200,
            }
        );
        let overridden_metadata = json!({"source": "synthetic_override"});
        let overridden_metadata = overridden_metadata
            .as_object()
            .ok_or_else(|| std::io::Error::other("synthetic metadata must be an object"))?;
        assert_eq!(
            repository.apply_ai_debt(
                7_000_000_000_016,
                Some(7_000_000_000_017),
                200,
                "invalid",
                "custom_debt",
                overridden_metadata,
            )?,
            BalancePairResult {
                user_balance: 300,
                chat_balance: -200,
            }
        );
        let debt_evidence = client.query_one(
            "SELECT COUNT(*), \
                COUNT(*) FILTER (WHERE metadata->>'source' = 'chat'), \
                COUNT(*) FILTER (WHERE metadata->>'source' = 'synthetic_override'), \
                COALESCE(SUM(amount), 0) \
             FROM credit_ledger WHERE user_id = $1",
            &[&7_000_000_000_016_i64],
        )?;
        assert_eq!(debt_evidence.get::<_, i64>(0), 2);
        assert_eq!(debt_evidence.get::<_, i64>(1), 1);
        assert_eq!(debt_evidence.get::<_, i64>(2), 1);
        assert_eq!(debt_evidence.get::<_, i64>(3), -1100);

        let first_database_url = database_url.clone();
        let second_database_url = database_url.clone();
        let first = std::thread::spawn(move || {
            BillingRepository::new(&first_database_url).apply_ai_debt(
                7_000_000_000_018,
                None,
                300,
                "user",
                "ai_settlement_debt",
                &serde_json::Map::new(),
            )
        });
        let second = std::thread::spawn(move || {
            BillingRepository::new(&second_database_url).apply_ai_debt(
                7_000_000_000_018,
                None,
                300,
                "user",
                "ai_settlement_debt",
                &serde_json::Map::new(),
            )
        });
        let mut concurrent_debt_balances = [
            first
                .join()
                .map_err(|_| std::io::Error::other("first AI debt thread panicked"))??
                .user_balance,
            second
                .join()
                .map_err(|_| std::io::Error::other("second AI debt thread panicked"))??
                .user_balance,
        ];
        concurrent_debt_balances.sort_unstable();
        assert_eq!(concurrent_debt_balances, [-600, -300]);

        client.batch_execute(
            "INSERT INTO credit_accounts (scope_type, scope_id, balance) VALUES \
                ('user', 7000000000019, 100), \
                ('chat', 7000000000020, 200) \
             ON CONFLICT (scope_type, scope_id) DO UPDATE SET balance = EXCLUDED.balance;",
        )?;
        let chat_refund_metadata = json!({"idempotency_key": "synthetic-refund-1"});
        let chat_refund_metadata = chat_refund_metadata
            .as_object()
            .ok_or_else(|| std::io::Error::other("synthetic metadata must be an object"))?;
        assert_eq!(
            repository.refund_ai_charge(
                7_000_000_000_019,
                Some(7_000_000_000_020),
                300,
                "chat",
                "ai_refund",
                chat_refund_metadata,
                Some("synthetic-refund-1"),
                "",
            )?,
            AiRefundResult {
                applied: true,
                reason: None,
                user_balance: 100,
                chat_balance: 500,
            }
        );
        assert_eq!(
            repository.refund_ai_charge(
                7_000_000_000_019,
                Some(7_000_000_000_020),
                300,
                "chat",
                "ai_refund",
                chat_refund_metadata,
                Some("synthetic-refund-1"),
                "",
            )?,
            AiRefundResult {
                applied: false,
                reason: None,
                user_balance: 100,
                chat_balance: 500,
            }
        );
        client.execute(
            "INSERT INTO credit_ledger \
                (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
             VALUES ('ai_settlement_result', $1, $1, $2, 0, $3)",
            &[
                &7_000_000_000_019_i64,
                &7_000_000_000_020_i64,
                &json!({"operation_id": "settled-operation"}),
            ],
        )?;
        let settled_metadata = json!({"operation_id": "settled-operation"});
        let settled_metadata = settled_metadata
            .as_object()
            .ok_or_else(|| std::io::Error::other("synthetic metadata must be an object"))?;
        assert_eq!(
            repository.refund_ai_charge(
                7_000_000_000_019,
                Some(7_000_000_000_020),
                200,
                "user",
                "ai_refund",
                settled_metadata,
                None,
                "settled-operation",
            )?,
            AiRefundResult {
                applied: false,
                reason: Some("operation_settled".to_owned()),
                user_balance: 100,
                chat_balance: 500,
            }
        );
        let user_refund_metadata = json!({"idempotency_key": "synthetic-refund-2"});
        let user_refund_metadata = user_refund_metadata
            .as_object()
            .ok_or_else(|| std::io::Error::other("synthetic metadata must be an object"))?;
        assert_eq!(
            repository.refund_ai_charge(
                7_000_000_000_019,
                Some(7_000_000_000_020),
                200,
                "invalid",
                "ai_refund",
                user_refund_metadata,
                Some("synthetic-refund-2"),
                "",
            )?,
            AiRefundResult {
                applied: true,
                reason: None,
                user_balance: 300,
                chat_balance: 500,
            }
        );
        let refund_evidence = client.query_one(
            "SELECT COUNT(*), COALESCE(SUM(amount), 0) \
             FROM credit_ledger WHERE user_id = $1 AND event_type = 'ai_refund'",
            &[&7_000_000_000_019_i64],
        )?;
        assert_eq!(refund_evidence.get::<_, i64>(0), 2);
        assert_eq!(refund_evidence.get::<_, i64>(1), 500);

        let first_database_url = database_url.clone();
        let second_database_url = database_url.clone();
        let first = std::thread::spawn(move || {
            let metadata = serde_json::Map::from_iter([(
                "idempotency_key".to_owned(),
                serde_json::Value::String("concurrent-refund".to_owned()),
            )]);
            BillingRepository::new(&first_database_url).refund_ai_charge(
                7_000_000_000_021,
                None,
                100,
                "user",
                "ai_refund",
                &metadata,
                Some("concurrent-refund"),
                "",
            )
        });
        let second = std::thread::spawn(move || {
            let metadata = serde_json::Map::from_iter([(
                "idempotency_key".to_owned(),
                serde_json::Value::String("concurrent-refund".to_owned()),
            )]);
            BillingRepository::new(&second_database_url).refund_ai_charge(
                7_000_000_000_021,
                None,
                100,
                "user",
                "ai_refund",
                &metadata,
                Some("concurrent-refund"),
                "",
            )
        });
        let concurrent_refund_results = [
            first
                .join()
                .map_err(|_| std::io::Error::other("first AI refund thread panicked"))??,
            second
                .join()
                .map_err(|_| std::io::Error::other("second AI refund thread panicked"))??,
        ];
        assert_eq!(
            concurrent_refund_results
                .iter()
                .filter(|result| result.applied)
                .count(),
            1
        );
        assert!(
            concurrent_refund_results
                .iter()
                .all(|result| result.user_balance == 100)
        );

        client.batch_execute(
            "INSERT INTO credit_accounts (scope_type, scope_id, balance) VALUES \
                ('user', 7000000000022, 500), \
                ('chat', 7000000000023, 700) \
             ON CONFLICT (scope_type, scope_id) DO UPDATE SET balance = EXCLUDED.balance;",
        )?;
        let charge_metadata = |key: &str, operation_id: &str| {
            serde_json::Map::from_iter([
                (
                    "idempotency_key".to_owned(),
                    serde_json::Value::String(key.to_owned()),
                ),
                (
                    "operation_id".to_owned(),
                    serde_json::Value::String(operation_id.to_owned()),
                ),
            ])
        };
        let first_charge_metadata = charge_metadata("synthetic-reserve-1", "operation-1");
        assert_eq!(
            repository.charge_ai_credits(
                7_000_000_000_022,
                Some(7_000_000_000_023),
                300,
                "ai_reserve",
                &first_charge_metadata,
                None,
                Some("synthetic-reserve-1"),
                "operation-1",
            )?,
            AiChargeResult {
                ok: true,
                applied: true,
                reason: None,
                source: Some("user".to_owned()),
                amount: 300,
                user_balance: 200,
                chat_balance: 700,
            }
        );
        assert_eq!(
            repository.charge_ai_credits(
                7_000_000_000_022,
                Some(7_000_000_000_023),
                300,
                "ai_reserve",
                &first_charge_metadata,
                None,
                Some("synthetic-reserve-1"),
                "operation-1",
            )?,
            AiChargeResult {
                ok: true,
                applied: false,
                reason: None,
                source: Some("user".to_owned()),
                amount: 300,
                user_balance: 200,
                chat_balance: 700,
            }
        );
        let second_charge_metadata = charge_metadata("synthetic-reserve-2", "operation-2");
        assert!(
            repository
                .charge_ai_credits(
                    7_000_000_000_022,
                    Some(7_000_000_000_023),
                    400,
                    "ai_reserve",
                    &second_charge_metadata,
                    None,
                    Some("synthetic-reserve-2"),
                    "operation-2",
                )?
                .source
                .is_some_and(|source| source == "chat")
        );
        let third_charge_metadata = charge_metadata("synthetic-reserve-3", "operation-3");
        assert_eq!(
            repository
                .charge_ai_credits(
                    7_000_000_000_022,
                    Some(7_000_000_000_023),
                    100,
                    "ai_reserve",
                    &third_charge_metadata,
                    Some("chat"),
                    Some("synthetic-reserve-3"),
                    "operation-3",
                )?
                .chat_balance,
            200
        );
        let insufficient_metadata = charge_metadata("synthetic-reserve-4", "operation-4");
        assert_eq!(
            repository.charge_ai_credits(
                7_000_000_000_022,
                Some(7_000_000_000_023),
                300,
                "ai_reserve",
                &insufficient_metadata,
                Some("user"),
                Some("synthetic-reserve-4"),
                "operation-4",
            )?,
            AiChargeResult {
                ok: false,
                applied: false,
                reason: None,
                source: None,
                amount: 0,
                user_balance: 200,
                chat_balance: 200,
            }
        );
        client.execute(
            "INSERT INTO credit_ledger \
                (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
             VALUES ('ai_settlement_result', $1, $1, $2, 0, $3)",
            &[
                &7_000_000_000_022_i64,
                &7_000_000_000_023_i64,
                &json!({"operation_id": "settled-charge-operation"}),
            ],
        )?;
        let settled_charge_metadata =
            charge_metadata("synthetic-reserve-settled", "settled-charge-operation");
        assert_eq!(
            repository
                .charge_ai_credits(
                    7_000_000_000_022,
                    Some(7_000_000_000_023),
                    10,
                    "ai_reserve",
                    &settled_charge_metadata,
                    None,
                    Some("synthetic-reserve-settled"),
                    "settled-charge-operation",
                )?
                .reason,
            Some("operation_settled".to_owned())
        );
        let refunded_charge_metadata = charge_metadata("synthetic-reserve-5", "operation-5");
        assert!(
            repository
                .charge_ai_credits(
                    7_000_000_000_022,
                    Some(7_000_000_000_023),
                    50,
                    "ai_reserve",
                    &refunded_charge_metadata,
                    Some("user"),
                    Some("synthetic-reserve-5"),
                    "operation-5",
                )?
                .applied
        );
        client.execute(
            "INSERT INTO credit_ledger \
                (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
             VALUES ('ai_refund', $1, $1, $2, 50, $3)",
            &[
                &7_000_000_000_022_i64,
                &7_000_000_000_023_i64,
                &json!({"settlement_id": "synthetic-reserve-5"}),
            ],
        )?;
        assert_eq!(
            repository
                .charge_ai_credits(
                    7_000_000_000_022,
                    Some(7_000_000_000_023),
                    50,
                    "ai_reserve",
                    &refunded_charge_metadata,
                    Some("user"),
                    Some("synthetic-reserve-5"),
                    "operation-5",
                )?
                .reason,
            Some("reservation_refunded".to_owned())
        );
        let charge_evidence = client.query_one(
            "SELECT COUNT(*), COALESCE(SUM(amount), 0) FROM credit_ledger \
             WHERE user_id = $1 AND event_type = 'ai_reserve'",
            &[&7_000_000_000_022_i64],
        )?;
        assert_eq!(charge_evidence.get::<_, i64>(0), 4);
        assert_eq!(charge_evidence.get::<_, i64>(1), -850);

        client.execute(
            "INSERT INTO credit_accounts (scope_type, scope_id, balance) \
             VALUES ('user', $1, 500) \
             ON CONFLICT (scope_type, scope_id) DO UPDATE SET balance = EXCLUDED.balance",
            &[&7_000_000_000_024_i64],
        )?;
        let first_database_url = database_url.clone();
        let second_database_url = database_url.clone();
        let first = std::thread::spawn(move || {
            let metadata = serde_json::Map::from_iter([(
                "idempotency_key".to_owned(),
                serde_json::Value::String("concurrent-charge".to_owned()),
            )]);
            BillingRepository::new(&first_database_url).charge_ai_credits(
                7_000_000_000_024,
                None,
                300,
                "ai_reserve",
                &metadata,
                None,
                Some("concurrent-charge"),
                "",
            )
        });
        let second = std::thread::spawn(move || {
            let metadata = serde_json::Map::from_iter([(
                "idempotency_key".to_owned(),
                serde_json::Value::String("concurrent-charge".to_owned()),
            )]);
            BillingRepository::new(&second_database_url).charge_ai_credits(
                7_000_000_000_024,
                None,
                300,
                "ai_reserve",
                &metadata,
                None,
                Some("concurrent-charge"),
                "",
            )
        });
        let concurrent_charge_results = [
            first
                .join()
                .map_err(|_| std::io::Error::other("first AI charge thread panicked"))??,
            second
                .join()
                .map_err(|_| std::io::Error::other("second AI charge thread panicked"))??,
        ];
        assert_eq!(
            concurrent_charge_results
                .iter()
                .filter(|result| result.applied)
                .count(),
            1
        );
        assert!(
            concurrent_charge_results
                .iter()
                .all(|result| result.ok && result.user_balance == 200)
        );

        let provider_metadata = json!({
            "operation_id": "synthetic-provider-operation",
            "segment_id": "segment-1",
            "segment": {"input_tokens": 12, "output_tokens": 34}
        });
        assert!(repository.record_ai_provider_usage(
            7_000_000_000_025,
            Some(7_000_000_000_023),
            &provider_metadata,
        )?);
        client.execute(
            "INSERT INTO credit_ledger \
                (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
             VALUES ('ai_reserve', $1, $1, NULL, 0, $2)",
            &[
                &7_000_000_000_045_i64,
                &json!({
                    "operation_id": "synthetic-zero-unsettled-operation",
                    "usage_tag": "synthetic-zero-unsettled-usage",
                    "source": "user"
                }),
            ],
        )?;
        assert!(!repository.record_ai_provider_usage(
            7_000_000_000_025,
            Some(7_000_000_000_023),
            &provider_metadata,
        )?);
        let first_database_url = database_url.clone();
        let second_database_url = database_url.clone();
        let first = std::thread::spawn(move || {
            let metadata = json!({
                "operation_id": "synthetic-provider-operation",
                "segment_id": "segment-2",
                "segment": {"input_tokens": 1}
            });
            BillingRepository::new(&first_database_url).record_ai_provider_usage(
                7_000_000_000_025,
                None,
                &metadata,
            )
        });
        let second = std::thread::spawn(move || {
            let metadata = json!({
                "operation_id": "synthetic-provider-operation",
                "segment_id": "segment-2",
                "segment": {"input_tokens": 1}
            });
            BillingRepository::new(&second_database_url).record_ai_provider_usage(
                7_000_000_000_025,
                None,
                &metadata,
            )
        });
        let concurrent_provider_results = [
            first
                .join()
                .map_err(|_| std::io::Error::other("first provider thread panicked"))??,
            second
                .join()
                .map_err(|_| std::io::Error::other("second provider thread panicked"))??,
        ];
        assert_eq!(
            concurrent_provider_results
                .iter()
                .filter(|inserted| **inserted)
                .count(),
            1
        );
        let provider_evidence = client.query_one(
            "SELECT COUNT(*), \
                COUNT(*) FILTER (WHERE metadata->'segment'->>'input_tokens' = '12') \
             FROM credit_ledger \
             WHERE user_id = $1 AND event_type = 'ai_provider_usage'",
            &[&7_000_000_000_025_i64],
        )?;
        assert_eq!(provider_evidence.get::<_, i64>(0), 2);
        assert_eq!(provider_evidence.get::<_, i64>(1), 1);
        let provider_segments = repository
            .list_ai_provider_segments(7_000_000_000_025, "synthetic-provider-operation")?;
        assert_eq!(provider_segments.len(), 2);
        assert_eq!(
            provider_segments[0]["input_tokens"],
            serde_json::Value::from(12)
        );
        assert!(repository.update_ai_provider_usage(
            "synthetic-provider-operation",
            "segment-1",
            &json!({"input_tokens": 99, "output_tokens": 100}),
        )?);
        assert!(!repository.update_ai_provider_usage(
            "synthetic-provider-operation",
            "missing-segment",
            &json!({"input_tokens": 0}),
        )?);
        let updated_provider_segments = repository
            .list_ai_provider_segments(7_000_000_000_025, "synthetic-provider-operation")?;
        assert_eq!(
            updated_provider_segments[0]["input_tokens"],
            serde_json::Value::from(99)
        );
        assert_eq!(
            updated_provider_segments[1]["input_tokens"],
            serde_json::Value::from(1)
        );

        client.batch_execute(
            "INSERT INTO credit_accounts (scope_type, scope_id, balance) VALUES \
                ('user', 7000000000026, 500), \
                ('user', 7000000000027, 500), \
                ('chat', 7000000000028, 700), \
                ('user', 7000000000029, 500), \
                ('user', 7000000000030, 0), \
                ('chat', 7000000000031, 0), \
                ('user', 7000000000032, 0) \
             ON CONFLICT (scope_type, scope_id) DO UPDATE SET balance = EXCLUDED.balance;",
        )?;
        let user_settlement_operation = "synthetic-user-settlement";
        let mut user_settlement_hold =
            charge_metadata("synthetic-user-settlement-hold", user_settlement_operation);
        user_settlement_hold.extend(serde_json::Map::from_iter([
            ("origin_chat_id".to_owned(), json!("synthetic-chat")),
            ("message_id".to_owned(), json!("synthetic-message")),
            ("usage_tag".to_owned(), json!("ai_response")),
        ]));
        assert!(
            repository
                .charge_ai_credits(
                    7_000_000_000_026,
                    None,
                    300,
                    "ai_reserve",
                    &user_settlement_hold,
                    None,
                    Some("synthetic-user-settlement-hold"),
                    user_settlement_operation,
                )?
                .applied
        );
        let caller_metadata = serde_json::Map::from_iter([
            (
                "source".to_owned(),
                serde_json::Value::String("caller-must-not-override".to_owned()),
            ),
            (
                "trace_id".to_owned(),
                serde_json::Value::String("synthetic-trace".to_owned()),
            ),
        ]);
        assert_eq!(
            repository.settle_ai_operation_once(
                7_000_000_000_026,
                None,
                user_settlement_operation,
                100,
                &caller_metadata,
            )?,
            AiSettlementResult {
                applied: true,
                source: Some("user".to_owned()),
                authorized_credit_units: Some(300),
                actual_credit_units: Some(100),
                refunded_credit_units: Some(200),
                debt_applied_credit_units: Some(0),
                user_balance: 400,
                chat_balance: 0,
            }
        );
        assert_eq!(
            repository.settle_ai_operation_once(
                7_000_000_000_026,
                None,
                user_settlement_operation,
                999,
                &serde_json::Map::new(),
            )?,
            AiSettlementResult {
                applied: false,
                source: None,
                authorized_credit_units: None,
                actual_credit_units: None,
                refunded_credit_units: None,
                debt_applied_credit_units: None,
                user_balance: 400,
                chat_balance: 0,
            }
        );
        let user_settlement_evidence = client.query_one(
            "SELECT \
                COUNT(*) FILTER (WHERE event_type = 'ai_refund'), \
                COUNT(*) FILTER (WHERE event_type = 'ai_settlement_result'), \
                COUNT(*) FILTER (WHERE metadata->>'source' = 'user'), \
                COUNT(*) FILTER (WHERE metadata->>'trace_id' = 'synthetic-trace'), \
                COUNT(*) FILTER (WHERE metadata->>'reserved_credit_units_total' = '300' \
                    AND metadata->>'settled_credit_units' = '100'), \
                COUNT(*) FILTER (WHERE event_type = 'ai_settlement_result' \
                    AND metadata->>'origin_chat_id' = 'synthetic-chat' \
                    AND metadata->>'message_id' = 'synthetic-message' \
                    AND metadata->>'usage_tag' = 'ai_response') \
             FROM credit_ledger WHERE user_id = $1 \
               AND metadata->>'operation_id' = $2",
            &[&7_000_000_000_026_i64, &user_settlement_operation],
        )?;
        assert_eq!(user_settlement_evidence.get::<_, i64>(0), 1);
        assert_eq!(user_settlement_evidence.get::<_, i64>(1), 1);
        assert_eq!(user_settlement_evidence.get::<_, i64>(2), 3);
        assert_eq!(user_settlement_evidence.get::<_, i64>(3), 2);
        assert_eq!(user_settlement_evidence.get::<_, i64>(4), 2);
        assert_eq!(user_settlement_evidence.get::<_, i64>(5), 1);

        let chat_settlement_operation = "synthetic-chat-settlement";
        let chat_settlement_hold =
            charge_metadata("synthetic-chat-settlement-hold", chat_settlement_operation);
        assert!(
            repository
                .charge_ai_credits(
                    7_000_000_000_027,
                    Some(7_000_000_000_028),
                    300,
                    "ai_reserve",
                    &chat_settlement_hold,
                    Some("chat"),
                    Some("synthetic-chat-settlement-hold"),
                    chat_settlement_operation,
                )?
                .applied
        );
        assert_eq!(
            repository.settle_ai_operation_once(
                7_000_000_000_027,
                Some(7_000_000_000_028),
                chat_settlement_operation,
                500,
                &serde_json::Map::new(),
            )?,
            AiSettlementResult {
                applied: true,
                source: Some("chat".to_owned()),
                authorized_credit_units: Some(300),
                actual_credit_units: Some(500),
                refunded_credit_units: Some(0),
                debt_applied_credit_units: Some(200),
                user_balance: 500,
                chat_balance: 200,
            }
        );
        let debt_evidence = client.query_one(
            "SELECT COUNT(*), COALESCE(SUM(amount), 0) FROM credit_ledger \
             WHERE user_id = $1 AND event_type = 'ai_settlement_debt' \
               AND metadata->>'operation_id' = $2 \
               AND metadata->>'source' = 'chat'",
            &[&7_000_000_000_027_i64, &chat_settlement_operation],
        )?;
        assert_eq!(debt_evidence.get::<_, i64>(0), 1);
        assert_eq!(debt_evidence.get::<_, i64>(1), -200);

        let concurrent_settlement_operation = "synthetic-concurrent-settlement";
        let concurrent_settlement_hold = charge_metadata(
            "synthetic-concurrent-settlement-hold",
            concurrent_settlement_operation,
        );
        assert!(
            repository
                .charge_ai_credits(
                    7_000_000_000_029,
                    None,
                    300,
                    "ai_reserve",
                    &concurrent_settlement_hold,
                    None,
                    Some("synthetic-concurrent-settlement-hold"),
                    concurrent_settlement_operation,
                )?
                .applied
        );
        let first_database_url = database_url.clone();
        let second_database_url = database_url.clone();
        let first = std::thread::spawn(move || {
            BillingRepository::new(&first_database_url).settle_ai_operation_once(
                7_000_000_000_029,
                None,
                "synthetic-concurrent-settlement",
                100,
                &serde_json::Map::new(),
            )
        });
        let second = std::thread::spawn(move || {
            BillingRepository::new(&second_database_url).settle_ai_operation_once(
                7_000_000_000_029,
                None,
                "synthetic-concurrent-settlement",
                100,
                &serde_json::Map::new(),
            )
        });
        let concurrent_settlement_results = [
            first
                .join()
                .map_err(|_| std::io::Error::other("first settlement thread panicked"))??,
            second
                .join()
                .map_err(|_| std::io::Error::other("second settlement thread panicked"))??,
        ];
        assert_eq!(
            concurrent_settlement_results
                .iter()
                .filter(|result| result.applied)
                .count(),
            1
        );
        assert!(
            concurrent_settlement_results
                .iter()
                .all(|result| result.user_balance == 400)
        );

        client.execute(
            "INSERT INTO credit_ledger \
                (event_type, actor_user_id, user_id, chat_id, amount, metadata) VALUES \
                ('ai_reserve', $1, $1, NULL, -1, $3), \
                ('ai_reserve', $1, $1, $2, -1, $4)",
            &[
                &7_000_000_000_030_i64,
                &7_000_000_000_031_i64,
                &json!({"operation_id": "synthetic-mixed-payer", "source": "user"}),
                &json!({"operation_id": "synthetic-mixed-payer", "source": "chat"}),
            ],
        )?;
        assert!(matches!(
            repository.settle_ai_operation_once(
                7_000_000_000_030,
                Some(7_000_000_000_031),
                "synthetic-mixed-payer",
                1,
                &serde_json::Map::new(),
            ),
            Err(BillingError::MultiplePayers)
        ));
        client.execute(
            "INSERT INTO credit_ledger \
                (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
             VALUES ('ai_reserve', $1, $1, $2, -1, $3)",
            &[
                &7_000_000_000_032_i64,
                &7_000_000_000_031_i64,
                &json!({"operation_id": "synthetic-missing-chat", "source": "chat"}),
            ],
        )?;
        assert!(matches!(
            repository.settle_ai_operation_once(
                7_000_000_000_032,
                None,
                "synthetic-missing-chat",
                1,
                &serde_json::Map::new(),
            ),
            Err(BillingError::ChatIdRequired)
        ));

        client.batch_execute(
            "INSERT INTO credit_accounts (scope_type, scope_id, balance) VALUES \
                ('user', 7000000000033, 200), \
                ('user', 7000000000034, 100), \
                ('chat', 7000000000035, 500), \
                ('user', 7000000000036, 0) \
             ON CONFLICT (scope_type, scope_id) DO UPDATE SET balance = EXCLUDED.balance;",
        )?;
        let legacy_metadata = serde_json::Map::from_iter([
            (
                "source".to_owned(),
                serde_json::Value::String("caller-override".to_owned()),
            ),
            (
                "trace_id".to_owned(),
                serde_json::Value::String("legacy-trace".to_owned()),
            ),
        ]);
        assert_eq!(
            repository.settle_legacy_ai_reservation_once(
                7_000_000_000_033,
                None,
                "user",
                300,
                100,
                "synthetic-legacy-user",
                &legacy_metadata,
            )?,
            LegacySettlementResult {
                applied: true,
                adjustment_credit_units: Some(200),
                user_balance: 400,
                chat_balance: 0,
            }
        );
        assert_eq!(
            repository.settle_legacy_ai_reservation_once(
                7_000_000_000_033,
                None,
                "user",
                999,
                0,
                "synthetic-legacy-user",
                &serde_json::Map::new(),
            )?,
            LegacySettlementResult {
                applied: false,
                adjustment_credit_units: None,
                user_balance: 400,
                chat_balance: 0,
            }
        );
        let legacy_user_evidence = client.query_one(
            "SELECT COUNT(*), COALESCE(SUM(amount), 0), \
                COUNT(*) FILTER (WHERE metadata->>'source' = 'caller-override' \
                    AND metadata->>'trace_id' = 'legacy-trace' \
                    AND metadata->>'reserved_credit_units' = '300') \
             FROM credit_ledger WHERE user_id = $1 \
               AND event_type = 'memory_compaction_settlement'",
            &[&7_000_000_000_033_i64],
        )?;
        assert_eq!(legacy_user_evidence.get::<_, i64>(0), 1);
        assert_eq!(legacy_user_evidence.get::<_, i64>(1), 200);
        assert_eq!(legacy_user_evidence.get::<_, i64>(2), 1);

        assert_eq!(
            repository.settle_legacy_ai_reservation_once(
                7_000_000_000_034,
                Some(7_000_000_000_035),
                "chat",
                300,
                500,
                "synthetic-legacy-chat",
                &serde_json::Map::new(),
            )?,
            LegacySettlementResult {
                applied: true,
                adjustment_credit_units: Some(-200),
                user_balance: 100,
                chat_balance: 300,
            }
        );
        assert!(matches!(
            repository.settle_legacy_ai_reservation_once(
                7_000_000_000_034,
                None,
                "chat",
                300,
                100,
                "synthetic-legacy-missing-chat",
                &serde_json::Map::new(),
            ),
            Err(BillingError::LegacyChatIdRequired)
        ));

        let first_database_url = database_url.clone();
        let second_database_url = database_url.clone();
        let first = std::thread::spawn(move || {
            BillingRepository::new(&first_database_url).settle_legacy_ai_reservation_once(
                7_000_000_000_036,
                None,
                "user",
                100,
                0,
                "synthetic-legacy-concurrent",
                &serde_json::Map::new(),
            )
        });
        let second = std::thread::spawn(move || {
            BillingRepository::new(&second_database_url).settle_legacy_ai_reservation_once(
                7_000_000_000_036,
                None,
                "user",
                100,
                0,
                "synthetic-legacy-concurrent",
                &serde_json::Map::new(),
            )
        });
        let concurrent_legacy_results = [
            first
                .join()
                .map_err(|_| std::io::Error::other("first legacy settlement thread panicked"))??,
            second
                .join()
                .map_err(|_| std::io::Error::other("second legacy settlement thread panicked"))??,
        ];
        assert_eq!(
            concurrent_legacy_results
                .iter()
                .filter(|result| result.applied)
                .count(),
            1
        );
        assert!(
            concurrent_legacy_results
                .iter()
                .all(|result| result.user_balance == 100)
        );

        let audit_metadata = serde_json::Map::from_iter([(
            "settlement_id".to_owned(),
            serde_json::Value::String("synthetic-audit-result".to_owned()),
        )]);
        assert!(repository.record_ai_settlement_result(
            7_000_000_000_037,
            Some(7_000_000_000_035),
            99,
            "ai_settlement_result",
            &audit_metadata,
        )?);
        assert!(!repository.record_ai_settlement_result(
            7_000_000_000_037,
            Some(7_000_000_000_035),
            99,
            "ai_settlement_result",
            &audit_metadata,
        )?);
        let audit_evidence = client.query_one(
            "SELECT COUNT(*), COALESCE(SUM(amount), 0), MIN(actor_user_id) \
             FROM credit_ledger WHERE user_id = $1 \
               AND event_type = 'ai_settlement_result' \
               AND metadata->>'settlement_id' = 'synthetic-audit-result'",
            &[&7_000_000_000_037_i64],
        )?;
        assert_eq!(audit_evidence.get::<_, i64>(0), 1);
        assert_eq!(audit_evidence.get::<_, i64>(1), 0);
        assert_eq!(audit_evidence.get::<_, Option<i64>>(2), Some(99));
        let recent_audit_results = repository.list_recent_ai_settlement_results(1)?;
        assert_eq!(recent_audit_results.len(), 1);
        assert_eq!(recent_audit_results[0].user_id, Some(7_000_000_000_037));
        assert_eq!(recent_audit_results[0].actor_user_id, Some(99));
        assert_eq!(recent_audit_results[0].amount, 0);
        assert_eq!(
            recent_audit_results[0].metadata["settlement_id"],
            serde_json::Value::String("synthetic-audit-result".to_owned())
        );
        assert!(!recent_audit_results[0].created_at.is_empty());

        client.execute(
            "INSERT INTO credit_ledger \
                (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
             VALUES ('ai_reserve', $1, $1, NULL, -300, $2)",
            &[
                &7_000_000_000_038_i64,
                &json!({
                    "operation_id": "synthetic-unsettled-operation",
                    "usage_tag": "synthetic-unsettled-usage",
                    "source": "user",
                    "trace_id": "unsettled-trace"
                }),
            ],
        )?;
        assert!(repository.record_ai_provider_usage(
            7_000_000_000_038,
            None,
            &json!({
                "operation_id": "synthetic-unsettled-operation",
                "segment_id": "unsettled-segment",
                "segment": {"input_tokens": 12}
            }),
        )?);
        client.execute(
            "INSERT INTO credit_ledger \
                (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
             VALUES ('ai_reserve', $1, $1, NULL, -100, $2)",
            &[
                &7_000_000_000_038_i64,
                &json!({
                    "operation_id": "synthetic-background-operation",
                    "usage_tag": "memory_compaction:synthetic",
                    "source": "user",
                    "background": true
                }),
            ],
        )?;
        client.execute(
            "INSERT INTO credit_ledger \
                (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
             VALUES ('ai_reserve', $1, $1, NULL, -100, $2)",
            &[
                &7_000_000_000_039_i64,
                &json!({
                    "operation_id": "synthetic-legacy-excluded-operation",
                    "usage_tag": "synthetic-legacy-excluded-usage",
                    "source": "user"
                }),
            ],
        )?;
        assert!(
            repository
                .settle_legacy_ai_reservation_once(
                    7_000_000_000_039,
                    None,
                    "user",
                    100,
                    100,
                    "synthetic-legacy-excluded-usage",
                    &serde_json::Map::new(),
                )?
                .applied
        );
        let unsettled_operations = repository.list_unsettled_ai_operations(500)?;
        let unsettled = unsettled_operations
            .iter()
            .find(|operation| operation.operation_id == "synthetic-unsettled-operation")
            .ok_or_else(|| std::io::Error::other("unsettled operation must be returned"))?;
        assert_eq!(unsettled.user_id, 7_000_000_000_038);
        assert_eq!(unsettled.authorized_credit_units, 300);
        assert_eq!(unsettled.source, "user");
        assert_eq!(unsettled.reserve_metadata["trace_id"], "unsettled-trace");
        assert_eq!(unsettled.segments.len(), 1);
        assert_eq!(unsettled.segments[0]["segment_id"], "unsettled-segment");
        assert!(!unsettled.created_at.is_empty());
        assert!(!unsettled.last_activity_at.is_empty());
        let zero_unsettled = unsettled_operations
            .iter()
            .find(|operation| operation.operation_id == "synthetic-zero-unsettled-operation")
            .ok_or_else(|| {
                std::io::Error::other("zero-cost unsettled operation must be returned")
            })?;
        assert_eq!(zero_unsettled.authorized_credit_units, 0);
        assert!(zero_unsettled.segments.is_empty());
        assert!(
            !unsettled_operations.iter().any(|operation| {
                operation.operation_id == "synthetic-legacy-excluded-operation"
            })
        );
        assert!(
            !unsettled_operations
                .iter()
                .any(|operation| operation.operation_id == "synthetic-chat-automation")
        );
        assert!(
            !unsettled_operations
                .iter()
                .any(|operation| { operation.operation_id == "synthetic-background-operation" })
        );

        client.execute(
            "INSERT INTO credit_ledger \
                (event_type, actor_user_id, user_id, chat_id, amount, metadata, created_at) \
             VALUES \
                ('ai_reconciliation_correction', $1, $1, NULL, 1, '{}', \
                    NOW() - INTERVAL '31 days'), \
                ('ai_reconciliation_correction', $1, $1, NULL, 1, '{}', NOW())",
            &[&7_000_000_000_040_i64],
        )?;
        assert_eq!(
            repository.purge_expired_ai_ledger_events(30)?,
            PurgeResult {
                deleted_rows: 1,
                retention_days: 30,
            }
        );
        assert_eq!(
            client
                .query_one(
                    "SELECT COUNT(*) FROM credit_ledger WHERE user_id = $1 \
                     AND event_type = 'ai_reconciliation_correction'",
                    &[&7_000_000_000_040_i64],
                )?
                .get::<_, i64>(0),
            1
        );

        let charge_history_metadata = serde_json::Map::from_iter([
            (
                "settlement_id".to_owned(),
                Value::String("synthetic:202:99:ai_response_base".to_owned()),
            ),
            ("origin_chat_id".to_owned(), Value::String("202".to_owned())),
            ("message_id".to_owned(), Value::String("99".to_owned())),
            (
                "charged_credit_units_total".to_owned(),
                Value::Number(123.into()),
            ),
        ]);
        assert!(repository.record_ai_settlement_result(
            7_000_000_000_041,
            Some(202),
            7_000_000_000_041,
            "ai_settlement_result",
            &charge_history_metadata,
        )?);
        assert!(repository.record_ai_settlement_result(
            7_000_000_000_041,
            Some(202),
            7_000_000_000_041,
            "ai_settlement_result",
            &serde_json::Map::from_iter([
                (
                    "settlement_id".to_owned(),
                    Value::String("synthetic:202:98:audio".to_owned()),
                ),
                ("origin_chat_id".to_owned(), Value::String("202".to_owned())),
                ("message_id".to_owned(), Value::String("98".to_owned())),
                (
                    "charged_credit_units_total".to_owned(),
                    Value::Number(0.into()),
                ),
            ]),
        )?);
        let finalized_charge_rows =
            repository.list_user_ai_charge_rows(7_000_000_000_041, None, "older", 21)?;
        assert_eq!(finalized_charge_rows.len(), 1);
        assert_eq!(finalized_charge_rows[0].group_key, "202:99");
        assert_eq!(
            finalized_charge_rows[0].metadata["charged_credit_units_total"],
            123
        );
        assert!(!finalized_charge_rows[0].created_at.is_empty());
        assert!(!finalized_charge_rows[0].group_created_at.is_empty());

        let history_user_id = 7_000_000_000_043_i64;
        let earlier_reserve = json!({
            "operation_id": "synthetic-history-earlier",
            "settlement_id": "synthetic-history-earlier:reserve",
            "usage_tag": "ai_response_base",
            "origin_chat_id": "404",
            "message_id": "1",
            "source": "user"
        });
        let later_reserve = json!({
            "operation_id": "synthetic-history-later",
            "settlement_id": "synthetic-history-later:reserve",
            "usage_tag": "ai_response_base",
            "origin_chat_id": "404",
            "message_id": "2",
            "source": "user"
        });
        let earlier_settlement = json!({
            "operation_id": "synthetic-history-earlier",
            "settlement_id": "synthetic-history-earlier:reserve",
            "usage_tag": "ai_response_base",
            "origin_chat_id": "404",
            "message_id": "1",
            "charged_credit_units_total": 1
        });
        let later_settlement = json!({
            "operation_id": "synthetic-history-later",
            "settlement_id": "synthetic-history-later:reserve",
            "usage_tag": "ai_response_base",
            "origin_chat_id": "404",
            "message_id": "2",
            "charged_credit_units_total": 1
        });
        client.execute(
            "INSERT INTO credit_ledger \
                (event_type, actor_user_id, user_id, chat_id, amount, metadata, created_at) \
             VALUES \
                ('ai_reserve', $1, $1, 404, -1, $2, NOW() - INTERVAL '2 hours'), \
                ('ai_reserve', $1, $1, 404, -1, $3, NOW() - INTERVAL '1 hour'), \
                ('ai_settlement_result', $1, $1, 404, 0, $4, NOW()), \
                ('ai_settlement_result', $1, $1, 404, 0, $5, NOW())",
            &[
                &history_user_id,
                &earlier_reserve,
                &later_reserve,
                &earlier_settlement,
                &later_settlement,
            ],
        )?;
        let reserve_times = client.query(
            "SELECT metadata->>'operation_id', created_at::text FROM credit_ledger \
             WHERE user_id = $1 AND event_type = 'ai_reserve' ORDER BY id",
            &[&history_user_id],
        )?;
        let history_rows =
            repository.list_user_ai_charge_rows(history_user_id, None, "older", 21)?;
        assert_eq!(history_rows.len(), 2);
        for reserve in reserve_times {
            let operation_id = reserve.get::<_, String>(0);
            let reserve_time = reserve.get::<_, String>(1);
            let history = history_rows
                .iter()
                .find(|row| row.metadata["operation_id"] == operation_id)
                .ok_or_else(|| std::io::Error::other("history operation must exist"))?;
            assert_eq!(history.group_created_at, reserve_time);
        }

        client.execute(
            "INSERT INTO credit_ledger \
                (event_type, actor_user_id, user_id, chat_id, amount, metadata) \
             VALUES ('ai_reserve', $1, $1, 303, -50, $2)",
            &[
                &7_000_000_000_042_i64,
                &json!({
                    "settlement_id": "synthetic:303:100:ai_response_base",
                    "origin_chat_id": "303",
                    "message_id": "100",
                    "source": "user"
                }),
            ],
        )?;
        let pending_charge_rows =
            repository.list_user_ai_charge_rows(7_000_000_000_042, None, "older", 21)?;
        assert_eq!(pending_charge_rows.len(), 1);
        assert_eq!(pending_charge_rows[0].event_type, "ai_reserve");
        assert_eq!(pending_charge_rows[0].group_key, "303:100");
        assert_eq!(pending_charge_rows[0].amount, -50);
        assert_eq!(pending_charge_rows[0].metadata["billing_pending"], true);
        assert_eq!(
            pending_charge_rows[0].metadata["charged_credit_units_total"],
            50
        );

        let synthetic_ids = [
            7_000_000_000_001_i64,
            7_000_000_000_002_i64,
            7_000_000_000_003_i64,
            7_000_000_000_004_i64,
            7_000_000_000_005_i64,
            7_000_000_000_006_i64,
            7_000_000_000_007_i64,
            7_000_000_000_008_i64,
            7_000_000_000_009_i64,
            7_000_000_000_010_i64,
            7_000_000_000_011_i64,
            7_000_000_000_012_i64,
            7_000_000_000_013_i64,
            7_000_000_000_014_i64,
            7_000_000_000_015_i64,
            7_000_000_000_016_i64,
            7_000_000_000_017_i64,
            7_000_000_000_018_i64,
            7_000_000_000_019_i64,
            7_000_000_000_020_i64,
            7_000_000_000_021_i64,
            7_000_000_000_022_i64,
            7_000_000_000_023_i64,
            7_000_000_000_024_i64,
            7_000_000_000_025_i64,
            7_000_000_000_026_i64,
            7_000_000_000_027_i64,
            7_000_000_000_028_i64,
            7_000_000_000_029_i64,
            7_000_000_000_030_i64,
            7_000_000_000_031_i64,
            7_000_000_000_032_i64,
            7_000_000_000_033_i64,
            7_000_000_000_034_i64,
            7_000_000_000_035_i64,
            7_000_000_000_036_i64,
            7_000_000_000_037_i64,
            7_000_000_000_038_i64,
            7_000_000_000_039_i64,
            7_000_000_000_040_i64,
            7_000_000_000_041_i64,
            7_000_000_000_042_i64,
            7_000_000_000_043_i64,
            7_000_000_000_044_i64,
        ];
        client.execute(
            "DELETE FROM star_payments WHERE user_id = ANY($1)",
            &[&&synthetic_ids[..]],
        )?;
        client.execute(
            "DELETE FROM credit_ledger WHERE user_id = ANY($1)",
            &[&&synthetic_ids[..]],
        )?;
        client.execute(
            "DELETE FROM credit_ledger WHERE chat_id = ANY($1)",
            &[&&synthetic_ids[..]],
        )?;
        client.execute(
            "DELETE FROM onboarding_grants WHERE user_id = ANY($1)",
            &[&&synthetic_ids[..]],
        )?;
        client.execute(
            "DELETE FROM credit_accounts WHERE scope_id = ANY($1)",
            &[&&synthetic_ids[..]],
        )?;
        Ok(())
    }
}
