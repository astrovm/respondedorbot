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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, PoisonError};

    use bot_adapters::billing_read::BillingRepository;
    use bot_adapters::lightning_charges::{
        LightningSettlement, NewLightningCharge, PendingLightningCharge,
    };
    use bot_adapters::opennode::{
        ChargeStatus, HttpResponse, NewCharge, OpenNodeRequest, OpenNodeTransport,
        TransportFailureKind,
    };
    use bot_core::lightning_topup::LightningInvoice;
    use bot_core::locale::Locale;
    use bot_core::telegram_actions::TelegramAction;
    use bot_core::telegram_input::{ChatId, MessageId};
    use bot_core::telegram_payments::default_billing_pack;

    use super::{
        LightningLedger, LightningPaymentWorker, LightningProvider, OpenNodeProvider,
        ProviderCheckout, opennode_provider,
    };
    use crate::background::BackgroundWorker;
    use crate::dispatcher::{ActionReceipt, ActionSink, LightningCheckout};

    type Log = Arc<Mutex<Vec<String>>>;

    fn entries(log: &Log) -> Vec<String> {
        log.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn push(log: &Log, entry: String) {
        log.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(entry);
    }

    #[derive(Default)]
    struct FakeProvider {
        created: Option<Result<LightningInvoice, String>>,
        statuses: HashMap<String, Result<ChargeStatus, String>>,
        log: Log,
    }

    impl LightningProvider for FakeProvider {
        fn create(&mut self, charge: &NewCharge) -> Result<LightningInvoice, String> {
            push(&self.log, format!("create {charge:?}"));
            self.created.take().unwrap_or(Err("no invoice".to_owned()))
        }

        fn status(&mut self, charge_id: &str) -> Result<ChargeStatus, String> {
            self.statuses
                .get(charge_id)
                .cloned()
                .unwrap_or(Ok(ChargeStatus::Pending))
        }
    }

    #[derive(Default)]
    struct FakeLedger {
        pending: Vec<PendingLightningCharge>,
        pending_error: Option<String>,
        settlements: HashMap<String, Option<LightningSettlement>>,
        record_error: Option<String>,
        log: Log,
    }

    impl LightningLedger for FakeLedger {
        fn record(&mut self, charge: &NewLightningCharge) -> Result<(), String> {
            push(&self.log, format!("record {charge:?}"));
            self.record_error.clone().map_or(Ok(()), Err)
        }

        fn attach_message(&mut self, charge_id: &str, message_id: i64) -> Result<(), String> {
            push(&self.log, format!("attach {charge_id} {message_id}"));
            Ok(())
        }

        fn pending(&mut self) -> Result<Vec<PendingLightningCharge>, String> {
            self.pending_error
                .clone()
                .map_or_else(|| Ok(self.pending.clone()), Err)
        }

        fn settle(&mut self, charge_id: &str) -> Result<Option<LightningSettlement>, String> {
            push(&self.log, format!("settle {charge_id}"));
            self.settlements
                .get(charge_id)
                .copied()
                .ok_or_else(|| "settle failed".to_owned())
        }

        fn close(&mut self, charge_id: &str) -> Result<(), String> {
            push(&self.log, format!("close {charge_id}"));
            Ok(())
        }
    }

    #[derive(Default)]
    struct FakeActions {
        sent: Arc<Mutex<Vec<TelegramAction>>>,
        fail: bool,
    }

    impl ActionSink for FakeActions {
        type Error = String;

        fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
            if self.fail {
                return Err("telegram down".to_owned());
            }
            self.sent
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(action);
            Ok(ActionReceipt { message_id: None })
        }
    }

    fn invoice() -> LightningInvoice {
        LightningInvoice {
            charge_id: "charge-1".to_owned(),
            payreq: "lnbc1synthetic".to_owned(),
            checkout_url: None,
            sats: Some(512),
        }
    }

    #[test]
    fn checkout_records_the_charge_before_returning_the_invoice() {
        let log = Log::default();
        let provider = FakeProvider {
            created: Some(Ok(invoice())),
            log: Arc::clone(&log),
            ..FakeProvider::default()
        };
        let ledger = FakeLedger {
            log: Arc::clone(&log),
            ..FakeLedger::default()
        };
        let mut checkout = ProviderCheckout::new(provider, ledger);
        let pack = default_billing_pack("p50");
        assert!(pack.is_some());
        let Some(pack) = pack else { return };
        assert_eq!(checkout.create(88, 88, &pack, Locale::En), Ok(invoice()));
        assert_eq!(checkout.attach_message("charge-1", 9), Ok(()));
        assert_eq!(
            entries(&log),
            [
                format!(
                    "create {:?}",
                    NewCharge {
                        usd_cents: 33,
                        description: "50 AI credits".to_owned(),
                        order_id: "88:p50".to_owned(),
                        ttl_minutes: 30,
                    }
                ),
                format!(
                    "record {:?}",
                    NewLightningCharge {
                        charge_id: "charge-1".to_owned(),
                        user_id: 88,
                        chat_id: 88,
                        pack_id: "p50".to_owned(),
                        usd_cents: 33,
                        credits_awarded: 5_000,
                        locale: "en".to_owned(),
                        ttl_minutes: 30,
                    }
                ),
                "attach charge-1 9".to_owned(),
            ]
        );
    }

    #[test]
    fn checkout_failures_hide_the_invoice() {
        let pack = default_billing_pack("p100");
        assert!(pack.is_some());
        let Some(pack) = pack else { return };
        let log = Log::default();
        let mut offline = ProviderCheckout::new(
            FakeProvider {
                log: Arc::clone(&log),
                ..FakeProvider::default()
            },
            FakeLedger {
                log: Arc::clone(&log),
                ..FakeLedger::default()
            },
        );
        assert_eq!(
            offline.create(88, 88, &pack, Locale::Es),
            Err("no invoice".to_owned())
        );
        let log = entries(&log);
        assert_eq!(log.len(), 1);
        assert!(log[0].contains("description: \"100 créditos de IA\""));

        let mut unrecorded = ProviderCheckout::new(
            FakeProvider {
                created: Some(Ok(invoice())),
                ..FakeProvider::default()
            },
            FakeLedger {
                record_error: Some("database down".to_owned()),
                ..FakeLedger::default()
            },
        );
        assert_eq!(
            unrecorded.create(88, 88, &pack, Locale::Es),
            Err("database down".to_owned())
        );
    }

    fn pending(
        charge_id: &str,
        locale: &str,
        message_id: Option<i64>,
        overdue: bool,
    ) -> PendingLightningCharge {
        PendingLightningCharge {
            charge_id: charge_id.to_owned(),
            chat_id: 88,
            message_id,
            locale: locale.to_owned(),
            overdue,
        }
    }

    #[test]
    fn worker_credits_paid_charges_once_and_closes_dead_ones() {
        let log = Log::default();
        let statuses = [
            ("paid-en", Ok(ChargeStatus::Paid)),
            ("paid-es", Ok(ChargeStatus::Paid)),
            ("already", Ok(ChargeStatus::Paid)),
            ("expired", Ok(ChargeStatus::Closed)),
            ("overdue", Ok(ChargeStatus::Pending)),
            ("waiting", Ok(ChargeStatus::Pending)),
        ]
        .into_iter()
        .map(|(id, status)| (id.to_owned(), status))
        .collect();
        let settled = Some(LightningSettlement {
            credits_awarded: 5_000,
            user_balance: 7_500,
        });
        let settlements = [
            ("paid-en", settled),
            ("paid-es", settled),
            ("already", None),
        ]
        .into_iter()
        .map(|(id, settlement)| (id.to_owned(), settlement))
        .collect();
        let ledger = FakeLedger {
            pending: vec![
                pending("paid-en", "en", Some(5), false),
                pending("paid-es", "es", None, false),
                pending("already", "es", None, false),
                pending("expired", "es", None, false),
                pending("overdue", "es", None, true),
                pending("waiting", "es", None, false),
            ],
            settlements,
            log: Arc::clone(&log),
            ..FakeLedger::default()
        };
        let actions = FakeActions::default();
        let sent = Arc::clone(&actions.sent);
        let provider = FakeProvider {
            statuses,
            ..FakeProvider::default()
        };
        let mut worker = LightningPaymentWorker::new(provider, ledger, actions);
        assert_eq!(worker.run_once(1), Ok(()));
        assert_eq!(
            entries(&log),
            [
                "settle paid-en",
                "settle paid-es",
                "settle already",
                "close expired",
                "close overdue",
            ]
        );
        let sent = sent.lock().unwrap_or_else(PoisonError::into_inner).clone();
        assert!(matches!(
            sent.as_slice(),
            [TelegramAction::SendMessage(english), TelegramAction::SendMessage(spanish)]
                if english.chat_id == ChatId(88)
                    && english.reply_to_message_id == Some(MessageId(5))
                    && english.text
                        == "Lightning payment received ⚡\n+50.00 credits\nPersonal balance: 75.00 credits"
                    && spanish.reply_to_message_id.is_none()
                    && spanish.text.starts_with("Pago Lightning recibido ⚡")
        ));
    }

    #[test]
    fn repeated_failures_are_reported_after_three_cycles() {
        let statuses = [
            ("lookup", Err("provider down".to_owned())),
            ("unsettled", Ok(ChargeStatus::Paid)),
            ("notice", Ok(ChargeStatus::Paid)),
        ]
        .into_iter()
        .map(|(id, status)| (id.to_owned(), status))
        .collect();
        let settlements = [(
            "notice".to_owned(),
            Some(LightningSettlement {
                credits_awarded: 100,
                user_balance: 100,
            }),
        )]
        .into_iter()
        .collect();
        let ledger = FakeLedger {
            pending: vec![
                pending("lookup", "es", None, false),
                pending("unsettled", "es", None, false),
                pending("notice", "es", None, false),
            ],
            settlements,
            ..FakeLedger::default()
        };
        let actions = FakeActions {
            fail: true,
            ..FakeActions::default()
        };
        let provider = FakeProvider {
            statuses,
            ..FakeProvider::default()
        };
        let mut worker = LightningPaymentWorker::new(provider, ledger, actions);
        assert_eq!(worker.run_once(1), Ok(()));
        assert_eq!(worker.run_once(2), Ok(()));
        assert_eq!(
            worker.run_once(3),
            Err(
                "3 consecutive Lightning poll cycles failed: lookup: provider down; \
                 unsettled: settle failed; notice: payment notice: telegram down"
                    .to_owned()
            )
        );

        let mut broken = LightningPaymentWorker::new(
            FakeProvider::default(),
            FakeLedger {
                pending_error: Some("database down".to_owned()),
                ..FakeLedger::default()
            },
            FakeActions::default(),
        );
        for _ in 0..2 {
            assert_eq!(broken.run_once(1), Ok(()));
        }
        assert_eq!(
            broken.run_once(1),
            Err(
                "3 consecutive Lightning poll cycles failed: pending charges: database down"
                    .to_owned()
            )
        );
        // A clean cycle resets the count.
        broken.ledger.pending_error = None;
        assert_eq!(broken.run_once(1), Ok(()));
        assert_eq!(broken.consecutive_failed_cycles, 0);
    }

    struct CannedTransport(Result<HttpResponse, TransportFailureKind>);

    impl OpenNodeTransport for CannedTransport {
        fn send(&self, _request: &OpenNodeRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.0.clone()
        }
    }

    fn canned(body: &str) -> OpenNodeProvider<CannedTransport> {
        OpenNodeProvider {
            transport: CannedTransport(Ok(HttpResponse {
                status_code: 200,
                body: body.to_owned(),
            })),
        }
    }

    #[test]
    fn opennode_provider_maps_charges_statuses_and_errors() {
        let mut provider = canned(
            r#"{"data":{"id":"c1","amount":10,"hosted_checkout_url":"https://x.test","lightning_invoice":{"payreq":"lnbc1"},"status":"paid"}}"#,
        );
        assert_eq!(
            provider.create(&NewCharge {
                usd_cents: 33,
                description: "d".to_owned(),
                order_id: "o".to_owned(),
                ttl_minutes: 30,
            }),
            Ok(LightningInvoice {
                charge_id: "c1".to_owned(),
                payreq: "lnbc1".to_owned(),
                checkout_url: Some("https://x.test".to_owned()),
                sats: Some(10),
            })
        );
        assert_eq!(provider.status("c1"), Ok(ChargeStatus::Paid));
        let mut broken = canned("{}");
        assert_eq!(
            broken.status("c1"),
            Err("OpenNode response is missing data".to_owned())
        );
        assert_eq!(
            broken
                .create(&NewCharge {
                    usd_cents: 1,
                    description: String::new(),
                    order_id: String::new(),
                    ttl_minutes: 10,
                })
                .map(|invoice| invoice.charge_id),
            Err("OpenNode response is missing data".to_owned())
        );
        assert!(opennode_provider("https://opennode.example.test", "synthetic-key").is_ok());
    }

    #[test]
    fn billing_repository_backs_the_lightning_ledger() {
        let Ok(url) = std::env::var("TEST_DATABASE_URL") else {
            return;
        };
        let schema = bot_adapters::billing_schema::BillingSchemaRepository::new(&url);
        assert!(schema.ensure_schema().is_ok());
        let mut ledger = BillingRepository::new(&url);
        let charge_id = format!("synthetic-ledger-{}", crate::test_env::fresh_op());
        let user_id = 7_000_000_000_401;
        assert_eq!(
            LightningLedger::record(
                &mut ledger,
                &NewLightningCharge {
                    charge_id: charge_id.clone(),
                    user_id,
                    chat_id: user_id,
                    pack_id: "p50".to_owned(),
                    usd_cents: 33,
                    credits_awarded: 5_000,
                    locale: "en".to_owned(),
                    ttl_minutes: 30,
                },
            ),
            Ok(())
        );
        assert_eq!(
            LightningLedger::attach_message(&mut ledger, &charge_id, 3),
            Ok(())
        );
        let pending = LightningLedger::pending(&mut ledger);
        assert!(pending.is_ok_and(|pending| {
            pending
                .iter()
                .any(|charge| charge.charge_id == charge_id && charge.message_id == Some(3))
        }));
        let settled = LightningLedger::settle(&mut ledger, &charge_id);
        assert!(matches!(
            settled,
            Ok(Some(LightningSettlement {
                credits_awarded: 5_000,
                ..
            }))
        ));
        assert_eq!(LightningLedger::close(&mut ledger, &charge_id), Ok(()));
    }
}
