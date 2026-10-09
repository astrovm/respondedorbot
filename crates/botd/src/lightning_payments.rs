//! Lightning top-ups through OpenNode: charge creation for the dispatcher and
//! a background poller that credits paid charges.

use bot_adapters::billing_read::BillingRepository;
use bot_adapters::lightning_charges::{
    LightningSettlement, NewLightningCharge, PendingLightningCharge,
};
use bot_adapters::opennode::{
    ChargeStatus, NewCharge, OpenNodeTransport, ReqwestOpenNodeTransport, charge_status,
    create_charge,
};
use bot_core::lightning_topup::{
    LIGHTNING_INVOICE_TTL_MINUTES, LightningInvoice, lightning_paid_reply, lightning_usd_cents,
};
use bot_core::locale::Locale;
use bot_core::telegram_actions::{SendMessage, TelegramAction};
use bot_core::telegram_input::{ChatId, MessageId};
use bot_core::telegram_payments::BillingPackTerms;

use crate::background::BackgroundWorker;
use crate::dispatcher::{ActionSink, LightningCheckout};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenNodeOptions {
    pub api_key: String,
    pub api_url: String,
}

/// Charges checked per poll, oldest first.
const POLL_BATCH: i64 = 50;
/// How long past expiry a charge is still polled before it is closed locally.
const EXPIRY_GRACE_MINUTES: i32 = 10;
const FAILURE_REPORT_THRESHOLD: usize = 3;

/// Charge creation, kept separate from storage so tests can fake either.
pub trait LightningProvider: Send {
    fn create(&mut self, charge: &NewCharge) -> Result<LightningInvoice, String>;

    fn status(&mut self, charge_id: &str) -> Result<ChargeStatus, String>;
}

pub struct OpenNodeProvider<T> {
    transport: T,
}

impl<T: OpenNodeTransport + Send> LightningProvider for OpenNodeProvider<T> {
    fn create(&mut self, charge: &NewCharge) -> Result<LightningInvoice, String> {
        create_charge(&self.transport, charge)
            .map(|charge| LightningInvoice {
                charge_id: charge.id,
                payreq: charge.payreq,
                checkout_url: charge.checkout_url,
                sats: charge.sats,
            })
            .map_err(|error| error.to_string())
    }

    fn status(&mut self, charge_id: &str) -> Result<ChargeStatus, String> {
        charge_status(&self.transport, charge_id).map_err(|error| error.to_string())
    }
}

/// Where Lightning charges are tracked until they are paid or closed.
pub trait LightningLedger: Send {
    fn record(&mut self, charge: &NewLightningCharge) -> Result<(), String>;

    fn attach_message(&mut self, charge_id: &str, message_id: i64) -> Result<(), String>;

    fn pending(&mut self) -> Result<Vec<PendingLightningCharge>, String>;

    fn settle(&mut self, charge_id: &str) -> Result<Option<LightningSettlement>, String>;

    fn close(&mut self, charge_id: &str) -> Result<(), String>;
}

impl LightningLedger for BillingRepository {
    fn record(&mut self, charge: &NewLightningCharge) -> Result<(), String> {
        self.record_lightning_charge(charge)
            .map_err(crate::error_text)
    }

    fn attach_message(&mut self, charge_id: &str, message_id: i64) -> Result<(), String> {
        self.attach_lightning_message(charge_id, message_id)
            .map_err(crate::error_text)
    }

    fn pending(&mut self) -> Result<Vec<PendingLightningCharge>, String> {
        self.pending_lightning_charges(POLL_BATCH, EXPIRY_GRACE_MINUTES)
            .map_err(crate::error_text)
    }

    fn settle(&mut self, charge_id: &str) -> Result<Option<LightningSettlement>, String> {
        self.settle_lightning_charge(charge_id)
            .map_err(crate::error_text)
    }

    fn close(&mut self, charge_id: &str) -> Result<(), String> {
        self.close_lightning_charge(charge_id)
            .map_err(crate::error_text)
    }
}

pub struct ProviderCheckout<P, L> {
    provider: P,
    ledger: L,
}

impl<P: LightningProvider, L: LightningLedger> ProviderCheckout<P, L> {
    pub const fn new(provider: P, ledger: L) -> Self {
        Self { provider, ledger }
    }
}

impl<P: LightningProvider, L: LightningLedger> LightningCheckout for ProviderCheckout<P, L> {
    /// The charge is stored before the invoice is shown, so every invoice a
    /// user can pay is one the poller will find.
    fn create(
        &mut self,
        user_id: i64,
        chat_id: i64,
        pack: &BillingPackTerms,
        locale: Locale,
    ) -> Result<LightningInvoice, String> {
        let usd_cents = lightning_usd_cents(pack);
        let credits = pack.credits_awarded / 100;
        let invoice = self.provider.create(&NewCharge {
            usd_cents,
            description: match locale {
                Locale::Es => format!("{credits} créditos de IA"),
                Locale::En => format!("{credits} AI credits"),
            },
            order_id: format!("{user_id}:{}", pack.id),
            ttl_minutes: LIGHTNING_INVOICE_TTL_MINUTES,
        })?;
        let int = |value: i64, field: &str| {
            i32::try_from(value).map_err(|_| format!("lightning {field} exceeds the integer range"))
        };
        self.ledger.record(&NewLightningCharge {
            charge_id: invoice.charge_id.clone(),
            user_id,
            chat_id,
            pack_id: pack.id.clone(),
            usd_cents: int(usd_cents, "price")?,
            credits_awarded: int(pack.credits_awarded, "credits")?,
            locale: locale.code().to_owned(),
            ttl_minutes: int(LIGHTNING_INVOICE_TTL_MINUTES.into(), "ttl")?,
        })?;
        Ok(invoice)
    }

    fn attach_message(&mut self, charge_id: &str, message_id: i64) -> Result<(), String> {
        self.ledger.attach_message(charge_id, message_id)
    }
}

/// Polls unpaid charges, credits paid ones once, and tells the buyer.
pub struct LightningPaymentWorker<P, L, A> {
    provider: P,
    ledger: L,
    actions: A,
    consecutive_failed_cycles: usize,
}

impl<P, L, A> LightningPaymentWorker<P, L, A> {
    pub const fn new(provider: P, ledger: L, actions: A) -> Self {
        Self {
            provider,
            ledger,
            actions,
            consecutive_failed_cycles: 0,
        }
    }
}

impl<P, L, A> LightningPaymentWorker<P, L, A>
where
    P: LightningProvider,
    L: LightningLedger,
    A: ActionSink + Send,
{
    fn poll(&mut self, charge: &PendingLightningCharge) -> Result<(), String> {
        let id = &charge.charge_id;
        match self.provider.status(id)? {
            ChargeStatus::Paid => {
                let Some(settlement) = self.ledger.settle(id)? else {
                    return Ok(());
                };
                let locale = if charge.locale == "en" {
                    Locale::En
                } else {
                    Locale::Es
                };
                let text = lightning_paid_reply(
                    settlement.credits_awarded,
                    settlement.user_balance,
                    locale,
                );
                let mut message = SendMessage::new(ChatId(charge.chat_id), &text);
                message.reply_to_message_id = charge.message_id.map(MessageId);
                // The credit is already committed; a failed notice only
                // loses the message, never the credits.
                self.actions
                    .execute(TelegramAction::SendMessage(message))
                    .map(|_receipt| ())
                    .map_err(|error| format!("payment notice: {error}"))
            }
            ChargeStatus::Closed => self.ledger.close(id),
            ChargeStatus::Pending if charge.overdue => self.ledger.close(id),
            ChargeStatus::Pending => Ok(()),
        }
    }
}

impl<P, L, A> BackgroundWorker for LightningPaymentWorker<P, L, A>
where
    P: LightningProvider + 'static,
    L: LightningLedger + 'static,
    A: ActionSink + Send + 'static,
{
    fn run_once(&mut self, _now_epoch_seconds: i64) -> Result<(), String> {
        let failures = match self.ledger.pending() {
            Ok(pending) => pending
                .iter()
                .filter_map(|charge| {
                    self.poll(charge)
                        .err()
                        .map(|error| format!("{}: {error}", charge.charge_id))
                })
                .collect(),
            Err(error) => vec![format!("pending charges: {error}")],
        };
        if failures.is_empty() {
            self.consecutive_failed_cycles = 0;
            return Ok(());
        }
        self.consecutive_failed_cycles = self.consecutive_failed_cycles.saturating_add(1);
        let message = failures.join("; ");
        if self.consecutive_failed_cycles >= FAILURE_REPORT_THRESHOLD {
            Err(format!(
                "{} consecutive Lightning poll cycles failed: {message}",
                self.consecutive_failed_cycles
            ))
        } else {
            eprintln!(
                "transient Lightning poll failure ({}/{FAILURE_REPORT_THRESHOLD}): {message}",
                self.consecutive_failed_cycles
            );
            Ok(())
        }
    }
}

pub fn opennode_provider(
    base_url: &str,
    api_key: &str,
) -> Result<OpenNodeProvider<ReqwestOpenNodeTransport>, String> {
    ReqwestOpenNodeTransport::new(base_url, api_key)
        .map(|transport| OpenNodeProvider { transport })
        .map_err(|error| format!("could not construct OpenNode transport: {error:?}"))
}
