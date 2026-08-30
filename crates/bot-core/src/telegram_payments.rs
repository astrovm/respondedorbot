//! Telegram Stars pre-checkout validation.

use serde::Serialize;
use serde_json::Value;
use thiserror::Error;

use crate::command_parsing::parse_command;
use crate::credit_units::{CreditUnits, format_credit_units};
use crate::locale::Locale;
use crate::telegram_actions::{
    InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice, SendMessage, TelegramAction,
};
use crate::telegram_input::{ChatId, MessageId, python_string, python_truthy};

const DEFAULT_BILLING_PACKS: [(&str, i64, i64); 6] = [
    ("p50", 25, 5_000),
    ("p100", 50, 10_000),
    ("p250", 125, 25_000),
    ("p500", 250, 50_000),
    ("p1000", 500, 100_000),
    ("p2500", 1_250, 250_000),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BillingPackTerms {
    pub id: String,
    pub xtr_amount: i64,
    pub credits_awarded: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StarPaymentRecord {
    pub charge_id: String,
    pub user_id: i64,
    pub pack_id: String,
    pub xtr_amount: i64,
    pub credits_awarded: i64,
    pub payload: String,
}

/// Return one production Telegram Stars pack using stored hundredth-credit units.
#[must_use]
pub fn default_billing_pack(pack_id: &str) -> Option<BillingPackTerms> {
    DEFAULT_BILLING_PACKS
        .iter()
        .find(|(id, _, _)| *id == pack_id)
        .map(|(id, xtr_amount, credits_awarded)| BillingPackTerms {
            id: (*id).to_owned(),
            xtr_amount: *xtr_amount,
            credits_awarded: *credits_awarded,
        })
}

#[must_use]
pub fn invoice_payload_locale(payload: &str) -> Option<&str> {
    let mut parts = payload.split(':');
    (parts.next() == Some("topup"))
        .then(|| {
            let _pack_id = parts.next()?;
            let _user_id = parts.next()?;
            parts.next()
        })
        .flatten()
}

fn topup_keyboard(locale: Locale) -> InlineKeyboardMarkup {
    InlineKeyboardMarkup {
        inline_keyboard: DEFAULT_BILLING_PACKS
            .iter()
            .map(|(id, xtr_amount, credits_awarded)| {
                let credits = format_credit_units(CreditUnits::new(*credits_awarded));
                vec![InlineKeyboardButton {
                    text: match locale {
                        Locale::Es => format!("{credits} créditos - {xtr_amount} ⭐"),
                        Locale::En => format!("{credits} credits - {xtr_amount} ⭐"),
                    },
                    url: None,
                    callback_data: Some(format!("topup:{id}")),
                }]
            })
            .collect(),
    }
}

#[must_use]
pub fn plan_topup_command(
    chat_id: ChatId,
    message_id: MessageId,
    message_text: &str,
    bot_name: &str,
    locale: Locale,
    chat_type: &str,
    billing_available: bool,
) -> Option<TelegramAction> {
    if parse_command(message_text, bot_name).command != "/topup" {
        return None;
    }
    let (text, keyboard) = if !billing_available {
        (
            match locale {
                Locale::Es => "el cobro de ia no está andando, avisale al admin",
                Locale::En => "AI billing is unavailable, please tell the admin",
            }
            .to_owned(),
            None,
        )
    } else if chat_type != "private" {
        let username = bot_name.trim().trim_start_matches('@');
        (
            match (locale, username.is_empty()) {
                (Locale::Es, false) => format!("la recarga va por privado, abrime en @{username}"),
                (Locale::En, false) => format!("top-ups are private, open @{username}"),
                (Locale::Es, true) => "la recarga va por privado, abrime en dm".to_owned(),
                (Locale::En, true) => "top-ups are private, open a DM with me".to_owned(),
            },
            None,
        )
    } else {
        (
            match locale {
                Locale::Es => "elegí cuánto querés cargar:",
                Locale::En => "choose how much you want to add:",
            }
            .to_owned(),
            Some(topup_keyboard(locale)),
        )
    };
    let mut message = SendMessage::new(chat_id, &text);
    message.reply_to_message_id = Some(message_id);
    message.reply_markup = keyboard;
    Some(TelegramAction::SendMessage(message))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BalanceCommandPlan {
    NotHandled,
    Reply(TelegramAction),
    Load {
        user_id: i64,
        chat_id: ChatId,
        is_group: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BalanceCommandContext {
    pub chat_id: ChatId,
    pub message_id: MessageId,
    pub user_id: Option<i64>,
    pub locale: Locale,
    pub is_group: bool,
    pub billing_available: bool,
}

fn reply(chat_id: ChatId, message_id: MessageId, text: &str) -> TelegramAction {
    let mut message = SendMessage::new(chat_id, text);
    message.reply_to_message_id = Some(message_id);
    TelegramAction::SendMessage(message)
}

#[must_use]
pub fn plan_balance_command(
    message_text: &str,
    bot_name: &str,
    context: BalanceCommandContext,
) -> BalanceCommandPlan {
    if parse_command(message_text, bot_name).command != "/balance" {
        return BalanceCommandPlan::NotHandled;
    }
    if !context.billing_available {
        return BalanceCommandPlan::Reply(reply(
            context.chat_id,
            context.message_id,
            match context.locale {
                Locale::Es => "el cobro de ia no está andando, avisale al admin",
                Locale::En => "AI billing is unavailable, please tell the admin",
            },
        ));
    }
    let Some(user_id) = context.user_id else {
        return BalanceCommandPlan::Reply(reply(
            context.chat_id,
            context.message_id,
            match context.locale {
                Locale::Es => "no te pude leer bien el usuario para ver los saldos",
                Locale::En => "I could not identify the user or chat to load the balances",
            },
        ));
    };
    BalanceCommandPlan::Load {
        user_id,
        chat_id: context.chat_id,
        is_group: context.is_group,
    }
}

#[must_use]
pub fn balance_reply(user_balance: i64, chat_balance: Option<i64>, locale: Locale) -> String {
    let user = format_credit_units(CreditUnits::new(user_balance));
    match (chat_balance, locale) {
        (None, Locale::Es) => {
            format!("tenés {user} créditos ia\nsi querés cargar más mandale /topup")
        }
        (None, Locale::En) => {
            format!("you have {user} AI credits\nuse /topup if you want to add more")
        }
        (Some(chat), Locale::Es) => {
            let chat = format_credit_units(CreditUnits::new(chat));
            format!(
                "saldos ia, maestro:\n- lo tuyo: {user}\n- lo del grupo: {chat}\nsi no alcanza lo tuyo, manoteo del grupo\nsi querés cargar más: /topup por privado\nsi querés pasarle al grupo: /transfer <monto>"
            )
        }
        (Some(chat), Locale::En) => {
            let chat = format_credit_units(CreditUnits::new(chat));
            format!(
                "AI balances:\n- yours: {user}\n- group: {chat}\nI use the group balance when yours runs out\nuse /topup in private to add more\nuse /transfer <amount> to move credits to the group"
            )
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopupCallbackPlan {
    Answer(Option<TelegramAction>),
    Invoice(Box<TopupInvoicePlan>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopupInvoicePlan {
    pub invoice: TelegramAction,
    pub success_answer: Option<TelegramAction>,
    pub failure_answer: Option<TelegramAction>,
}

fn callback_answer(
    callback_id: Option<&str>,
    text: Option<String>,
    show_alert: bool,
) -> Option<TelegramAction> {
    callback_id.map(|callback_id| TelegramAction::AnswerCallback {
        callback_id: callback_id.to_owned(),
        text,
        show_alert,
    })
}

fn invoice_action(
    chat_id: ChatId,
    user_id: i64,
    pack: &BillingPackTerms,
    locale: Locale,
) -> TelegramAction {
    let credits = format_credit_units(CreditUnits::new(pack.credits_awarded));
    let (title, description, label) = match locale {
        Locale::Es => (
            format!("Pack IA {credits} créditos"),
            format!("Recarga de {credits} créditos para mensajes IA"),
            format!("{credits} créditos IA"),
        ),
        Locale::En => (
            format!("{credits} AI credit pack"),
            format!("Add {credits} credits for AI messages"),
            format!("{credits} AI credits"),
        ),
    };
    TelegramAction::SendInvoice {
        chat_id,
        title,
        description,
        payload: format!("topup:{}:{user_id}:{}", pack.id, locale.code()),
        currency: "XTR".to_owned(),
        prices: vec![LabeledPrice {
            label,
            amount: pack.xtr_amount,
        }],
    }
}

#[must_use]
pub fn plan_topup_callback(
    callback_id: Option<&str>,
    data: &str,
    chat_id: ChatId,
    chat_type: &str,
    user_id: Option<i64>,
    billing_available: bool,
    locale: Locale,
) -> TopupCallbackPlan {
    let alert = |text: &str| {
        TopupCallbackPlan::Answer(callback_answer(callback_id, Some(text.to_owned()), true))
    };
    if !billing_available {
        return alert(match locale {
            Locale::Es => "el cobro de ia no está andando, avisale al admin",
            Locale::En => "AI billing is unavailable, please tell the admin",
        });
    }
    if chat_type != "private" {
        return alert(match locale {
            Locale::Es => "cargá por privado, maestro",
            Locale::En => "open this in a private chat",
        });
    }
    let pack = data
        .split_once(':')
        .filter(|(prefix, _)| *prefix == "topup")
        .and_then(|(_, pack_id)| default_billing_pack(pack_id));
    let Some(pack) = pack else {
        return alert(match locale {
            Locale::Es => "ese pack es fruta, elegí otro",
            Locale::En => "that credit pack is invalid",
        });
    };
    let Some(user_id) = user_id else {
        return TopupCallbackPlan::Answer(callback_answer(callback_id, None, false));
    };
    TopupCallbackPlan::Invoice(Box::new(TopupInvoicePlan {
        invoice: invoice_action(chat_id, user_id, &pack, locale),
        success_answer: callback_answer(
            callback_id,
            Some(
                match locale {
                    Locale::Es => "listo, te dejé la factura",
                    Locale::En => "invoice ready",
                }
                .to_owned(),
            ),
            false,
        ),
        failure_answer: callback_answer(
            callback_id,
            Some(
                match locale {
                    Locale::Es => "no pude armar la factura, probá de nuevo",
                    Locale::En => "I could not create the invoice, try again",
                }
                .to_owned(),
            ),
            true,
        ),
    }))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PreCheckoutDecision {
    Ignore,
    BillingUnavailable { query_id: String },
    InvalidUser { query_id: String },
    InvalidPayment { query_id: String },
    Approve { query_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SuccessfulPaymentDecision {
    Ignore,
    BillingUnavailable {
        chat_id: String,
    },
    InvalidPayment {
        chat_id: String,
        user_id: i64,
        currency: String,
        payload: String,
        total_amount: i64,
        charge_id: String,
    },
    Record {
        chat_id: String,
        user_id: i64,
        charge_id: String,
        pack_id: String,
        xtr_amount: i64,
        credits_awarded: i64,
        payload: String,
    },
}

/// Build the exact user-visible reply after an idempotent Stars ledger write.
#[must_use]
pub fn successful_payment_reply(
    credits_awarded: i64,
    user_balance: i64,
    inserted: bool,
    locale: Locale,
) -> String {
    let credits = format_credit_units(CreditUnits::new(credits_awarded));
    let balance = format_credit_units(CreditUnits::new(user_balance));
    match (inserted, locale) {
        (true, Locale::Es) => format!(
            "listo, te cargué {credits} créditos\nahora te quedaron {balance}\nsi querés mandarle al grupo: /transfer <monto>"
        ),
        (true, Locale::En) => format!(
            "added {credits} credits\nyour balance is now {balance}\nuse /transfer <amount> to fund a group"
        ),
        (false, Locale::Es) => {
            format!("ese pago ya estaba cargado, no rompas las bolas\nte quedaron {balance}")
        }
        (false, Locale::En) => {
            format!("this payment was already credited\nyour balance is {balance}")
        }
    }
}

/// Convert a validated payment decision into the typed PostgreSQL write input.
#[must_use]
pub fn payment_record(decision: &SuccessfulPaymentDecision) -> Option<StarPaymentRecord> {
    let SuccessfulPaymentDecision::Record {
        user_id,
        charge_id,
        pack_id,
        xtr_amount,
        credits_awarded,
        payload,
        ..
    } = decision
    else {
        return None;
    };
    Some(StarPaymentRecord {
        charge_id: charge_id.clone(),
        user_id: *user_id,
        pack_id: pack_id.clone(),
        xtr_amount: *xtr_amount,
        credits_awarded: *credits_awarded,
        payload: payload.clone(),
    })
}

/// Evaluate a successful payment against the production Stars pack catalog.
pub fn evaluate_default_successful_payment(
    message: &Value,
    billing_available: bool,
) -> Result<SuccessfulPaymentDecision, PaymentValidationError> {
    let payload = message
        .as_object()
        .and_then(|message| message.get("successful_payment"))
        .and_then(Value::as_object)
        .and_then(|payment| payment.get("invoice_payload"))
        .map_or_else(String::new, python_string);
    let pack = parse_topup_payload(&payload)
        .0
        .as_deref()
        .and_then(default_billing_pack);
    evaluate_successful_payment(message, billing_available, pack.as_ref())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PaymentValidationError {
    #[error("Telegram pre-checkout query must be an object")]
    InvalidQuery,
    #[error("Telegram pre-checkout sender is malformed")]
    InvalidSender,
    #[error("Telegram payment message, chat, or payment payload is malformed")]
    InvalidPaymentMessage,
}

fn optional_truthy_string(value: Option<&Value>) -> Option<String> {
    value
        .filter(|value| python_truthy(value))
        .map(python_string)
}

fn string_or_empty(value: Option<&Value>) -> String {
    optional_truthy_string(value).unwrap_or_default()
}

fn strict_python_int(value: Option<&Value>) -> Option<i64> {
    value.and_then(|value| python_string(value).parse().ok())
}

fn sender(
    query: &serde_json::Map<String, Value>,
) -> Result<&serde_json::Map<String, Value>, PaymentValidationError> {
    match query.get("from") {
        Some(Value::Object(sender)) => Ok(sender),
        Some(value) if python_truthy(value) => Err(PaymentValidationError::InvalidSender),
        Some(_) | None => {
            static EMPTY: std::sync::OnceLock<serde_json::Map<String, Value>> =
                std::sync::OnceLock::new();
            Ok(EMPTY.get_or_init(serde_json::Map::new))
        }
    }
}

#[must_use]
pub fn parse_topup_payload(payload: &str) -> (Option<String>, Option<i64>) {
    if payload.is_empty() {
        return (None, None);
    }
    let parts = payload.split(':').collect::<Vec<_>>();
    if parts.len() < 2 || parts[0] != "topup" {
        return (None, None);
    }
    let user_id = parts.get(2).and_then(|value| value.parse().ok());
    (Some(parts[1].to_owned()), user_id)
}

pub fn evaluate_pre_checkout(
    query: &Value,
    billing_available: bool,
    expected_pack: Option<&BillingPackTerms>,
) -> Result<PreCheckoutDecision, PaymentValidationError> {
    let query = query
        .as_object()
        .ok_or(PaymentValidationError::InvalidQuery)?;
    let Some(query_id) = optional_truthy_string(query.get("id")) else {
        return Ok(PreCheckoutDecision::Ignore);
    };
    if !billing_available {
        return Ok(PreCheckoutDecision::BillingUnavailable { query_id });
    }
    let user_id = strict_python_int(sender(query)?.get("id"));
    let Some(user_id) = user_id else {
        return Ok(PreCheckoutDecision::InvalidUser { query_id });
    };
    let payload = string_or_empty(query.get("invoice_payload"));
    let (pack_id, payload_user_id) = parse_topup_payload(&payload);
    let total_amount = strict_python_int(query.get("total_amount")).unwrap_or(-1);
    let currency = string_or_empty(query.get("currency"));
    let valid = expected_pack.is_some_and(|pack| {
        pack_id.as_deref() == Some(pack.id.as_str())
            && currency == "XTR"
            && total_amount == pack.xtr_amount
            && payload_user_id.is_none_or(|payload_user_id| payload_user_id == user_id)
    });
    Ok(if valid {
        PreCheckoutDecision::Approve { query_id }
    } else {
        PreCheckoutDecision::InvalidPayment { query_id }
    })
}

/// Validate and localize one native pre-checkout answer without performing I/O.
pub fn plan_pre_checkout(
    query: &Value,
    billing_available: bool,
    locale: Locale,
) -> Result<Option<TelegramAction>, PaymentValidationError> {
    let payload = query
        .as_object()
        .and_then(|query| query.get("invoice_payload"))
        .map_or_else(String::new, python_string);
    let pack = parse_topup_payload(&payload)
        .0
        .as_deref()
        .and_then(default_billing_pack);
    let decision = evaluate_pre_checkout(query, billing_available, pack.as_ref())?;
    let action = match decision {
        PreCheckoutDecision::Ignore => None,
        PreCheckoutDecision::Approve { query_id } => Some(TelegramAction::AnswerPreCheckout {
            query_id,
            ok: true,
            error_message: None,
        }),
        PreCheckoutDecision::BillingUnavailable { query_id } => {
            Some(TelegramAction::AnswerPreCheckout {
                query_id,
                ok: false,
                error_message: Some(match locale {
                    Locale::Es => "el cobro de ia no está andando, avisale al admin".to_owned(),
                    Locale::En => "AI billing is unavailable, please tell the admin".to_owned(),
                }),
            })
        }
        PreCheckoutDecision::InvalidUser { query_id } => Some(TelegramAction::AnswerPreCheckout {
            query_id,
            ok: false,
            error_message: Some(match locale {
                Locale::Es => "tu usuario vino medio roto para cobrar".to_owned(),
                Locale::En => "I could not identify your user for this payment".to_owned(),
            }),
        }),
        PreCheckoutDecision::InvalidPayment { query_id } => {
            Some(TelegramAction::AnswerPreCheckout {
                query_id,
                ok: false,
                error_message: Some(match locale {
                    Locale::Es => "ese pago vino raro y no te lo pude validar".to_owned(),
                    Locale::En => "I could not validate this payment".to_owned(),
                }),
            })
        }
    };
    Ok(action)
}

fn object_or_empty<'a>(
    value: Option<&'a Value>,
    empty: &'a serde_json::Map<String, Value>,
) -> Result<&'a serde_json::Map<String, Value>, PaymentValidationError> {
    match value {
        Some(Value::Object(value)) => Ok(value),
        Some(value) if python_truthy(value) => Err(PaymentValidationError::InvalidPaymentMessage),
        Some(_) | None => Ok(empty),
    }
}

pub fn evaluate_successful_payment(
    message: &Value,
    billing_available: bool,
    expected_pack: Option<&BillingPackTerms>,
) -> Result<SuccessfulPaymentDecision, PaymentValidationError> {
    let message = message
        .as_object()
        .ok_or(PaymentValidationError::InvalidPaymentMessage)?;
    let empty = serde_json::Map::new();
    let chat = object_or_empty(message.get("chat"), &empty)?;
    let Some(chat_id_value) = chat.get("id").filter(|value| !value.is_null()) else {
        return Ok(SuccessfulPaymentDecision::Ignore);
    };
    let chat_id = python_string(chat_id_value);
    if !billing_available {
        return Ok(SuccessfulPaymentDecision::BillingUnavailable { chat_id });
    }
    let Some(user_id) = message
        .get("from")
        .and_then(Value::as_object)
        .and_then(|user| user.get("id"))
        .and_then(crate::telegram_input::normalize_numeric_id)
    else {
        return Ok(SuccessfulPaymentDecision::Ignore);
    };
    let payment = object_or_empty(message.get("successful_payment"), &empty)?;
    let currency = string_or_empty(payment.get("currency"));
    let payload = string_or_empty(payment.get("invoice_payload"));
    let charge_id = string_or_empty(payment.get("telegram_payment_charge_id"));
    let total_amount = strict_python_int(payment.get("total_amount")).unwrap_or(-1);
    let (pack_id, payload_user_id) = parse_topup_payload(&payload);
    let valid = !charge_id.is_empty()
        && expected_pack.is_some_and(|pack| {
            pack_id.as_deref() == Some(pack.id.as_str())
                && currency == "XTR"
                && total_amount == pack.xtr_amount
                && payload_user_id.is_none_or(|payload_user_id| payload_user_id == user_id)
        });
    let Some(pack) = expected_pack.filter(|_| valid) else {
        return Ok(SuccessfulPaymentDecision::InvalidPayment {
            chat_id,
            user_id,
            currency,
            payload,
            total_amount,
            charge_id,
        });
    };
    Ok(SuccessfulPaymentDecision::Record {
        chat_id,
        user_id,
        charge_id,
        pack_id: pack.id.clone(),
        xtr_amount: pack.xtr_amount,
        credits_awarded: pack.credits_awarded,
        payload,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        BalanceCommandContext, BalanceCommandPlan, BillingPackTerms, PaymentValidationError,
        PreCheckoutDecision, StarPaymentRecord, SuccessfulPaymentDecision, TopupCallbackPlan,
        balance_reply, default_billing_pack, evaluate_default_successful_payment,
        evaluate_pre_checkout, evaluate_successful_payment, invoice_payload_locale,
        parse_topup_payload, payment_record, plan_balance_command, plan_pre_checkout,
        plan_topup_callback, plan_topup_command, successful_payment_reply,
    };
    use crate::locale::Locale;
    use crate::telegram_actions::{LabeledPrice, TelegramAction};
    use crate::telegram_input::{ChatId, MessageId};

    fn pack() -> BillingPackTerms {
        BillingPackTerms {
            id: "p50".to_owned(),
            xtr_amount: 25,
            credits_awarded: 5_000,
        }
    }

    #[test]
    fn production_pack_catalog_matches_the_python_credit_scale() {
        assert_eq!(default_billing_pack("p50"), Some(pack()));
        assert_eq!(
            default_billing_pack("p2500"),
            Some(BillingPackTerms {
                id: "p2500".to_owned(),
                xtr_amount: 1_250,
                credits_awarded: 250_000,
            })
        );
        assert_eq!(default_billing_pack("missing"), None);
        assert_eq!(invoice_payload_locale("topup:p50:42:en"), Some("en"));
        assert_eq!(invoice_payload_locale("topup:p50"), None);
        assert_eq!(invoice_payload_locale("other:p50:42:en"), None);
    }

    #[test]
    fn topup_command_plans_private_catalog_group_redirect_and_unavailable_reply() {
        let private = plan_topup_command(
            ChatId(42),
            MessageId(7),
            "/topup@mybot",
            "@mybot",
            Locale::En,
            "private",
            true,
        );
        let Some(TelegramAction::SendMessage(private)) = private else {
            return;
        };
        assert_eq!(private.text, "choose how much you want to add:");
        assert_eq!(private.reply_to_message_id, Some(MessageId(7)));
        let keyboard =
            private
                .reply_markup
                .unwrap_or(crate::telegram_actions::InlineKeyboardMarkup {
                    inline_keyboard: Vec::new(),
                });
        assert_eq!(keyboard.inline_keyboard.len(), 6);
        assert_eq!(
            keyboard.inline_keyboard[0][0].callback_data.as_deref(),
            Some("topup:p50")
        );
        assert_eq!(keyboard.inline_keyboard[0][0].text, "50.00 credits - 25 ⭐");

        for (chat_type, available, bot_name, locale, expected) in [
            (
                "group",
                true,
                "@mybot",
                Locale::Es,
                "la recarga va por privado, abrime en @mybot",
            ),
            (
                "group",
                true,
                "",
                Locale::En,
                "top-ups are private, open a DM with me",
            ),
            (
                "private",
                false,
                "@mybot",
                Locale::En,
                "AI billing is unavailable, please tell the admin",
            ),
        ] {
            let Some(TelegramAction::SendMessage(message)) = plan_topup_command(
                ChatId(42),
                MessageId(7),
                "/topup",
                bot_name,
                locale,
                chat_type,
                available,
            ) else {
                return;
            };
            assert_eq!(message.text, expected);
            assert!(message.reply_markup.is_none());
        }
        assert_eq!(
            plan_topup_command(
                ChatId(42),
                MessageId(7),
                "/balance",
                "@mybot",
                Locale::Es,
                "private",
                true,
            ),
            None
        );
    }

    #[test]
    fn balance_command_plans_external_loads_and_early_replies() {
        assert_eq!(
            plan_balance_command(
                "/balance@mybot",
                "@mybot",
                BalanceCommandContext {
                    chat_id: ChatId(-42),
                    message_id: MessageId(7),
                    user_id: Some(88),
                    locale: Locale::En,
                    is_group: true,
                    billing_available: true,
                },
            ),
            BalanceCommandPlan::Load {
                user_id: 88,
                chat_id: ChatId(-42),
                is_group: true,
            }
        );
        assert_eq!(
            plan_balance_command(
                "/other",
                "@mybot",
                BalanceCommandContext {
                    chat_id: ChatId(42),
                    message_id: MessageId(7),
                    user_id: Some(88),
                    locale: Locale::Es,
                    is_group: false,
                    billing_available: true,
                },
            ),
            BalanceCommandPlan::NotHandled
        );
        for (user_id, available, locale, expected) in [
            (
                Some(88),
                false,
                Locale::En,
                "AI billing is unavailable, please tell the admin",
            ),
            (
                None,
                true,
                Locale::Es,
                "no te pude leer bien el usuario para ver los saldos",
            ),
        ] {
            let BalanceCommandPlan::Reply(TelegramAction::SendMessage(message)) =
                plan_balance_command(
                    "/balance",
                    "@mybot",
                    BalanceCommandContext {
                        chat_id: ChatId(42),
                        message_id: MessageId(7),
                        user_id,
                        locale,
                        is_group: false,
                        billing_available: available,
                    },
                )
            else {
                return;
            };
            assert_eq!(message.text, expected);
            assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
        }
    }

    #[test]
    fn balance_replies_match_private_and_group_credit_formatting() {
        assert_eq!(
            balance_reply(4_200, None, Locale::Es),
            "tenés 42.00 créditos ia\nsi querés cargar más mandale /topup"
        );
        assert_eq!(
            balance_reply(4_200, None, Locale::En),
            "you have 42.00 AI credits\nuse /topup if you want to add more"
        );
        assert_eq!(
            balance_reply(3_000, Some(12_000), Locale::Es),
            "saldos ia, maestro:\n- lo tuyo: 30.00\n- lo del grupo: 120.00\nsi no alcanza lo tuyo, manoteo del grupo\nsi querés cargar más: /topup por privado\nsi querés pasarle al grupo: /transfer <monto>"
        );
        assert_eq!(
            balance_reply(3_000, Some(12_000), Locale::En),
            "AI balances:\n- yours: 30.00\n- group: 120.00\nI use the group balance when yours runs out\nuse /topup in private to add more\nuse /transfer <amount> to move credits to the group"
        );
    }

    #[test]
    fn topup_callback_plans_invoice_and_all_guard_answers() {
        let plan = plan_topup_callback(
            Some("callback-1"),
            "topup:p50",
            ChatId(42),
            "private",
            Some(42),
            true,
            Locale::En,
        );
        assert_eq!(
            plan,
            TopupCallbackPlan::Invoice(Box::new(super::TopupInvoicePlan {
                invoice: TelegramAction::SendInvoice {
                    chat_id: ChatId(42),
                    title: "50.00 AI credit pack".to_owned(),
                    description: "Add 50.00 credits for AI messages".to_owned(),
                    payload: "topup:p50:42:en".to_owned(),
                    currency: "XTR".to_owned(),
                    prices: vec![LabeledPrice {
                        label: "50.00 AI credits".to_owned(),
                        amount: 25,
                    }],
                },
                success_answer: Some(TelegramAction::AnswerCallback {
                    callback_id: "callback-1".to_owned(),
                    text: Some("invoice ready".to_owned()),
                    show_alert: false,
                }),
                failure_answer: Some(TelegramAction::AnswerCallback {
                    callback_id: "callback-1".to_owned(),
                    text: Some("I could not create the invoice, try again".to_owned()),
                    show_alert: true,
                }),
            }))
        );

        for (data, chat_type, user_id, available, locale, expected, alert) in [
            (
                "topup:p50",
                "private",
                Some(42),
                false,
                Locale::Es,
                Some("el cobro de ia no está andando, avisale al admin"),
                true,
            ),
            (
                "topup:p50",
                "group",
                Some(42),
                true,
                Locale::En,
                Some("open this in a private chat"),
                true,
            ),
            (
                "topup:missing",
                "private",
                Some(42),
                true,
                Locale::Es,
                Some("ese pack es fruta, elegí otro"),
                true,
            ),
            ("topup:p50", "private", None, true, Locale::En, None, false),
        ] {
            assert_eq!(
                plan_topup_callback(
                    Some("callback-2"),
                    data,
                    ChatId(42),
                    chat_type,
                    user_id,
                    available,
                    locale,
                ),
                TopupCallbackPlan::Answer(Some(TelegramAction::AnswerCallback {
                    callback_id: "callback-2".to_owned(),
                    text: expected.map(ToOwned::to_owned),
                    show_alert: alert,
                }))
            );
        }
        assert_eq!(
            plan_topup_callback(
                None,
                "topup:p50",
                ChatId(42),
                "private",
                None,
                true,
                Locale::En,
            ),
            TopupCallbackPlan::Answer(None)
        );
    }

    #[test]
    fn native_pre_checkout_planner_localizes_every_answer_kind() {
        let valid = json!({
            "id":"checkout-1",
            "from":{"id":42},
            "invoice_payload":"topup:p50:42:en",
            "currency":"XTR",
            "total_amount":25
        });
        assert_eq!(
            plan_pre_checkout(&valid, true, Locale::En),
            Ok(Some(TelegramAction::AnswerPreCheckout {
                query_id: "checkout-1".to_owned(),
                ok: true,
                error_message: None,
            }))
        );
        assert_eq!(plan_pre_checkout(&json!({}), true, Locale::Es), Ok(None));

        for (query, available, locale, expected) in [
            (
                json!({"id":"checkout-2"}),
                false,
                Locale::Es,
                "el cobro de ia no está andando, avisale al admin",
            ),
            (
                json!({"id":"checkout-3"}),
                true,
                Locale::En,
                "I could not identify your user for this payment",
            ),
            (
                json!({
                    "id":"checkout-4",
                    "from":{"id":42},
                    "invoice_payload":"topup:missing:42",
                    "currency":"XTR",
                    "total_amount":25
                }),
                true,
                Locale::Es,
                "ese pago vino raro y no te lo pude validar",
            ),
        ] {
            assert_eq!(
                plan_pre_checkout(&query, available, locale),
                Ok(Some(TelegramAction::AnswerPreCheckout {
                    query_id: query["id"].as_str().unwrap_or_default().to_owned(),
                    ok: false,
                    error_message: Some(expected.to_owned()),
                }))
            );
        }
    }

    #[test]
    fn default_successful_payment_evaluation_and_record_use_production_terms() {
        let message = json!({
            "chat":{"id":42},
            "from":{"id":42},
            "successful_payment":{
                "currency":"XTR",
                "invoice_payload":"topup:p100:42:es",
                "telegram_payment_charge_id":"charge-1",
                "total_amount":50
            }
        });
        let decision = evaluate_default_successful_payment(&message, true);
        assert_eq!(
            decision,
            Ok(SuccessfulPaymentDecision::Record {
                chat_id: "42".to_owned(),
                user_id: 42,
                charge_id: "charge-1".to_owned(),
                pack_id: "p100".to_owned(),
                xtr_amount: 50,
                credits_awarded: 10_000,
                payload: "topup:p100:42:es".to_owned(),
            })
        );
        assert_eq!(
            decision.as_ref().ok().and_then(payment_record),
            Some(StarPaymentRecord {
                charge_id: "charge-1".to_owned(),
                user_id: 42,
                pack_id: "p100".to_owned(),
                xtr_amount: 50,
                credits_awarded: 10_000,
                payload: "topup:p100:42:es".to_owned(),
            })
        );
        assert_eq!(payment_record(&SuccessfulPaymentDecision::Ignore), None);
    }

    #[test]
    fn successful_payment_replies_preserve_exact_credit_format_and_locale() {
        assert_eq!(
            successful_payment_reply(5_000, 5_300, true, Locale::Es),
            "listo, te cargué 50.00 créditos\nahora te quedaron 53.00\nsi querés mandarle al grupo: /transfer <monto>"
        );
        assert_eq!(
            successful_payment_reply(5_000, 5_300, true, Locale::En),
            "added 50.00 credits\nyour balance is now 53.00\nuse /transfer <amount> to fund a group"
        );
        assert_eq!(
            successful_payment_reply(5_000, 5_300, false, Locale::Es),
            "ese pago ya estaba cargado, no rompas las bolas\nte quedaron 53.00"
        );
        assert_eq!(
            successful_payment_reply(5_000, 5_300, false, Locale::En),
            "this payment was already credited\nyour balance is 53.00"
        );
    }

    #[test]
    fn parses_current_legacy_and_invalid_payloads() {
        assert_eq!(
            parse_topup_payload("topup:p50:42:en"),
            (Some("p50".to_owned()), Some(42))
        );
        assert_eq!(
            parse_topup_payload("topup:p50"),
            (Some("p50".to_owned()), None)
        );
        assert_eq!(
            parse_topup_payload("topup:p50:not-a-user"),
            (Some("p50".to_owned()), None)
        );
        assert_eq!(parse_topup_payload(""), (None, None));
        assert_eq!(parse_topup_payload("other:p50"), (None, None));
    }

    #[test]
    fn approves_exact_current_and_legacy_invoices() {
        for payload in ["topup:p50:42:en", "topup:p50"] {
            assert_eq!(
                evaluate_pre_checkout(
                    &json!({
                        "id":"checkout-1",
                        "from":{"id":"42"},
                        "invoice_payload":payload,
                        "currency":"XTR",
                        "total_amount":"25"
                    }),
                    true,
                    Some(&pack()),
                ),
                Ok(PreCheckoutDecision::Approve {
                    query_id: "checkout-1".to_owned()
                })
            );
        }
    }

    #[test]
    fn decision_order_matches_query_identity_and_billing_availability() {
        assert_eq!(
            evaluate_pre_checkout(&json!({}), true, Some(&pack())),
            Ok(PreCheckoutDecision::Ignore)
        );
        assert_eq!(
            evaluate_pre_checkout(&json!({"id":"checkout"}), false, None),
            Ok(PreCheckoutDecision::BillingUnavailable {
                query_id: "checkout".to_owned()
            })
        );
        assert_eq!(
            evaluate_pre_checkout(&json!({"id":"checkout"}), true, Some(&pack())),
            Ok(PreCheckoutDecision::InvalidUser {
                query_id: "checkout".to_owned()
            })
        );
    }

    #[test]
    fn rejects_every_payment_mismatch() {
        let cases = [
            ("topup:other:42", "XTR", 25, Some(pack())),
            ("topup:p50:43", "XTR", 25, Some(pack())),
            ("topup:p50:42", "USD", 25, Some(pack())),
            ("topup:p50:42", "XTR", 24, Some(pack())),
            ("topup:p50:42", "XTR", 25, None),
        ];
        for (payload, currency, total_amount, pack) in cases {
            assert_eq!(
                evaluate_pre_checkout(
                    &json!({
                        "id":"checkout",
                        "from":{"id":42},
                        "invoice_payload":payload,
                        "currency":currency,
                        "total_amount":total_amount
                    }),
                    true,
                    pack.as_ref(),
                ),
                Ok(PreCheckoutDecision::InvalidPayment {
                    query_id: "checkout".to_owned()
                })
            );
        }
    }

    #[test]
    fn malformed_boundaries_fall_back_without_approving() {
        assert_eq!(
            evaluate_pre_checkout(&json!([]), true, Some(&pack())),
            Err(PaymentValidationError::InvalidQuery)
        );
        assert_eq!(
            evaluate_pre_checkout(&json!({"id":"checkout","from":"bad"}), true, Some(&pack())),
            Err(PaymentValidationError::InvalidSender)
        );
        assert_eq!(
            evaluate_pre_checkout(
                &json!({
                    "id":"checkout",
                    "from":{"id":42},
                    "invoice_payload":"topup:p50:42",
                    "currency":"XTR",
                    "total_amount":"not-a-number"
                }),
                true,
                Some(&pack())
            ),
            Ok(PreCheckoutDecision::InvalidPayment {
                query_id: "checkout".to_owned()
            })
        );
    }

    #[test]
    fn successful_payment_decisions_preserve_early_exit_order() {
        assert_eq!(
            evaluate_successful_payment(&json!({}), true, Some(&pack())),
            Ok(SuccessfulPaymentDecision::Ignore)
        );
        assert_eq!(
            evaluate_successful_payment(&json!({"chat":{"id":42}}), false, None),
            Ok(SuccessfulPaymentDecision::BillingUnavailable {
                chat_id: "42".to_owned()
            })
        );
        assert_eq!(
            evaluate_successful_payment(&json!({"chat":{"id":42}}), true, Some(&pack())),
            Ok(SuccessfulPaymentDecision::Ignore)
        );
    }

    #[test]
    fn successful_payment_returns_typed_record_inputs() {
        assert_eq!(
            evaluate_successful_payment(
                &json!({
                    "chat":{"id":100},
                    "from":{"id":42},
                    "successful_payment":{
                        "currency":"XTR",
                        "invoice_payload":"topup:p50:42:es",
                        "telegram_payment_charge_id":"charge-1",
                        "total_amount":25
                    }
                }),
                true,
                Some(&pack())
            ),
            Ok(SuccessfulPaymentDecision::Record {
                chat_id: "100".to_owned(),
                user_id: 42,
                charge_id: "charge-1".to_owned(),
                pack_id: "p50".to_owned(),
                xtr_amount: 25,
                credits_awarded: 5_000,
                payload: "topup:p50:42:es".to_owned(),
            })
        );
    }

    #[test]
    fn successful_payment_rejects_invalid_terms_with_audit_fields() {
        assert_eq!(
            evaluate_successful_payment(
                &json!({
                    "chat":{"id":100},
                    "from":{"id":42},
                    "successful_payment":{
                        "currency":"USD",
                        "invoice_payload":"topup:p50:99",
                        "telegram_payment_charge_id":"",
                        "total_amount":"bad"
                    }
                }),
                true,
                Some(&pack())
            ),
            Ok(SuccessfulPaymentDecision::InvalidPayment {
                chat_id: "100".to_owned(),
                user_id: 42,
                currency: "USD".to_owned(),
                payload: "topup:p50:99".to_owned(),
                total_amount: -1,
                charge_id: String::new(),
            })
        );
        assert_eq!(
            evaluate_successful_payment(
                &json!({"chat":"bad","from":{"id":42}}),
                true,
                Some(&pack())
            ),
            Err(PaymentValidationError::InvalidPaymentMessage)
        );
    }
}
