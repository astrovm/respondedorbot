//! PostgreSQL billing repository used during the incremental writer cutover.

use native_tls::TlsConnector;
use postgres::{Client, Transaction, error::SqlState};
use postgres_native_tls::MakeTlsConnector;
use serde::Serialize;
use serde_json::{Map, Value, json};
use thiserror::Error;

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
            let metadata = billing_metadata(ledger_source, metadata);
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
                    metadata,
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
                    metadata,
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
                    COUNT(DISTINCT metadata->>'source'), MIN(metadata->>'source') \
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
            let settlement_metadata = settlement_metadata(
                metadata,
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
                WHERE pending.authorized > 0 OR EXISTS ( \
                    SELECT 1 FROM credit_ledger AS usage \
                    WHERE usage.event_type = 'ai_provider_usage' \
                      AND usage.metadata->>'operation_id' = pending.operation_id \
                ) \
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
        let connector = TlsConnector::builder().build()?;
        Ok(Client::connect(
            &self.database_url,
            MakeTlsConnector::new(connector),
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
    use serde_json::json;

    use super::{
        AiChargeResult, AiRefundResult, AiSettlementResult, BalancePairResult, BillingError,
        BillingRepository, BillingScope, ChatAiChargeResult, LegacySettlementResult,
        OnboardingGrantResult, StarPaymentResult, TransferResult,
    };

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
    fn reads_existing_and_missing_balances_when_test_postgres_is_available()
    -> Result<(), Box<dyn std::error::Error>> {
        let Ok(database_url) = std::env::var("TEST_POSTGRES_URL") else {
            return Ok(());
        };
        let connector = TlsConnector::builder().build()?;
        let mut client = Client::connect(&database_url, MakeTlsConnector::new(connector))?;
        client.batch_execute(
            "CREATE TABLE IF NOT EXISTS credit_accounts (\
                scope_type TEXT NOT NULL, \
                scope_id BIGINT NOT NULL, \
                balance INTEGER NOT NULL, \
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), \
                PRIMARY KEY (scope_type, scope_id)\
            ); \
            CREATE TABLE IF NOT EXISTS onboarding_grants (\
                user_id BIGINT PRIMARY KEY, \
                credits INTEGER NOT NULL, \
                granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()\
            ); \
            CREATE TABLE IF NOT EXISTS star_payments (\
                telegram_payment_charge_id TEXT PRIMARY KEY, \
                user_id BIGINT NOT NULL, \
                pack_id TEXT NOT NULL, \
                xtr_amount INTEGER NOT NULL, \
                credits_awarded INTEGER NOT NULL, \
                payload TEXT, \
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()\
            ); \
            CREATE TABLE IF NOT EXISTS credit_ledger (\
                id BIGSERIAL PRIMARY KEY, \
                event_type TEXT NOT NULL, \
                actor_user_id BIGINT, \
                user_id BIGINT, \
                chat_id BIGINT, \
                amount INTEGER NOT NULL, \
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb, \
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()\
            ); \
            CREATE UNIQUE INDEX IF NOT EXISTS idx_credit_ledger_unique_ai_provider_segment \
            ON credit_ledger ((metadata->>'operation_id'), (metadata->>'segment_id')) \
            WHERE event_type = 'ai_provider_usage' \
              AND metadata ? 'operation_id' AND metadata ? 'segment_id'; \
            CREATE UNIQUE INDEX IF NOT EXISTS idx_credit_ledger_unique_ai_settlement \
            ON credit_ledger (user_id, (metadata->>'settlement_id')) \
            WHERE event_type = 'ai_settlement_result' \
              AND metadata ? 'settlement_id'; \
            INSERT INTO credit_accounts (scope_type, scope_id, balance) \
            VALUES ('user', 7000000000001, 1234) \
            ON CONFLICT (scope_type, scope_id) DO UPDATE SET balance = EXCLUDED.balance;",
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
        let user_settlement_hold =
            charge_metadata("synthetic-user-settlement-hold", user_settlement_operation);
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
                    AND metadata->>'settled_credit_units' = '100') \
             FROM credit_ledger WHERE user_id = $1 \
               AND metadata->>'operation_id' = $2",
            &[&7_000_000_000_026_i64, &user_settlement_operation],
        )?;
        assert_eq!(user_settlement_evidence.get::<_, i64>(0), 1);
        assert_eq!(user_settlement_evidence.get::<_, i64>(1), 1);
        assert_eq!(user_settlement_evidence.get::<_, i64>(2), 3);
        assert_eq!(user_settlement_evidence.get::<_, i64>(3), 2);
        assert_eq!(user_settlement_evidence.get::<_, i64>(4), 2);

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
