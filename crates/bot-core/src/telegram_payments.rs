//! Telegram Stars pre-checkout validation.

use serde::Serialize;
use serde_json::Value;
use thiserror::Error;

use crate::command_parsing::parse_command;
use crate::credit_units::{CreditUnits, display_credit_units};
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

/// Smallest and largest top-up a buyer can pick, in whole credits.
pub const MIN_TOPUP_CREDITS: i64 = 10;
pub const MAX_TOPUP_CREDITS: i64 = 100_000;
/// What `/topup` starts on.
pub const DEFAULT_TOPUP_CREDITS: i64 = 100;
const QUICK_TOPUP_CREDITS: [i64; 4] = [50, 100, 500, 1_000];
/// A typical AI message costs 0.21 credits with memory upkeep (production,
/// 30 days); rounded up so the estimate doesn't overpromise.
const ESTIMATED_CREDIT_UNITS_PER_MESSAGE: i64 = 25;

/// Any whole number of credits in range, at the same 2 credits per Star as
/// the fixed packs. Odd amounts round the Stars up.
#[must_use]
pub fn custom_billing_pack(credits: i64) -> Option<BillingPackTerms> {
    (MIN_TOPUP_CREDITS..=MAX_TOPUP_CREDITS)
        .contains(&credits)
        .then(|| topup_terms(credits))
}

fn topup_terms(credits: i64) -> BillingPackTerms {
    BillingPackTerms {
        id: format!("c{credits}"),
        xtr_amount: (credits + 1) / 2,
        credits_awarded: credits * 100,
    }
}

/// Return one production Telegram Stars pack using stored hundredth-credit
/// units: a fixed pack (`p50`) or any amount (`c300`).
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
        .or_else(|| {
            pack_id
                .strip_prefix('c')
                .and_then(|credits| credits.parse().ok())
                .and_then(custom_billing_pack)
                // One id per amount, so "c050" can't stand in for "c50".
                .filter(|pack| pack.id == pack_id)
        })
}

/// Every fixed pack, smallest first.
pub fn billing_packs() -> impl Iterator<Item = BillingPackTerms> {
    DEFAULT_BILLING_PACKS
        .iter()
        .map(|(id, xtr_amount, credits_awarded)| BillingPackTerms {
            id: (*id).to_owned(),
            xtr_amount: *xtr_amount,
            credits_awarded: *credits_awarded,
        })
}

/// A typed amount: digits, with optional thousands separators.
#[must_use]
pub fn parse_topup_amount(text: &str) -> Option<i64> {
    let digits = text
        .trim()
        .chars()
        .filter(|character| !matches!(character, '.' | ',' | ' '))
        .collect::<String>();
    (!digits.is_empty() && digits.chars().all(|character| character.is_ascii_digit()))
        .then(|| digits.parse().ok())
        .flatten()
}

/// Bigger steps for bigger amounts, so a few taps reach any of them.
const fn topup_step(credits: i64) -> i64 {
    match credits {
        ..100 => 10,
        100..1_000 => 50,
        1_000..10_000 => 500,
        _ => 5_000,
    }
}

const fn more_credits(credits: i64) -> i64 {
    let next = credits + topup_step(credits);
    if next > MAX_TOPUP_CREDITS {
        MAX_TOPUP_CREDITS
    } else {
        next
    }
}

const fn fewer_credits(credits: i64) -> i64 {
    let previous = credits - topup_step(credits - 1);
    if previous < MIN_TOPUP_CREDITS {
        MIN_TOPUP_CREDITS
    } else {
        previous
    }
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

/// Pack sizes are whole credits, so buttons drop the ",00".
pub(crate) fn whole_credits(units: i64) -> String {
    let formatted = display_credit_units(CreditUnits::new(units));
    formatted
        .strip_suffix(".00")
        .map_or_else(|| formatted.clone(), ToOwned::to_owned)
}

/// First line of the top-up menu, which also marks a reply to it as a
/// typed amount.
#[must_use]
pub const fn topup_menu_title(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => "Cargar créditos",
        Locale::En => "Add credits",
    }
}

/// The top-up menu for one amount: change it with − and +, a quick pick or a
/// typed number, then pay with Telegram Stars (card, Google Pay, Apple Pay)
/// or Lightning.
#[must_use]
pub fn topup_menu(
    locale: Locale,
    credits: i64,
    lightning_available: bool,
) -> (String, InlineKeyboardMarkup) {
    use crate::menu_ui::button;
    let credits = credits.clamp(MIN_TOPUP_CREDITS, MAX_TOPUP_CREDITS);
    let pack = topup_terms(credits);
    let amount = whole_credits(pack.credits_awarded);
    let messages = crate::output_format::readable_number(
        &(pack.credits_awarded / ESTIMATED_CREDIT_UNITS_PER_MESSAGE).to_string(),
    );
    let text = match locale {
        Locale::Es => format!(
            "{}\n\n{amount} créditos ≈ {messages} mensajes de IA\nCambiá el monto o mandame el número",
            topup_menu_title(locale)
        ),
        Locale::En => format!(
            "{}\n\n{amount} credits ≈ {messages} AI messages\nChange the amount or send me a number",
            topup_menu_title(locale)
        ),
    };
    let amount_label = match locale {
        Locale::Es => format!("{amount} créditos"),
        Locale::En => format!("{amount} credits"),
    };
    let stars = crate::output_format::readable_number(&pack.xtr_amount.to_string());
    let mut rows = vec![
        vec![
            button("−", format!("topup:amt:{}", fewer_credits(credits))),
            button(amount_label, format!("topup:amt:{credits}")),
            button("+", format!("topup:amt:{}", more_credits(credits))),
        ],
        QUICK_TOPUP_CREDITS
            .iter()
            .map(|quick| {
                let label = whole_credits(quick * 100);
                button(
                    if *quick == credits {
                        format!("✓ {label}")
                    } else {
                        label
                    },
                    format!("topup:amt:{quick}"),
                )
            })
            .collect(),
        vec![button(
            format!("💳 Telegram  {stars} ⭐"),
            format!("topup:{}", pack.id),
        )],
    ];
    if lightning_available {
        let price =
            crate::lightning_topup::format_usd(crate::lightning_topup::lightning_usd_cents(&pack));
        rows.push(vec![button(
            format!("⚡ Lightning  {price}"),
            format!(
                "{}{}",
                crate::lightning_topup::LIGHTNING_PACK_PREFIX,
                pack.id
            ),
        )]);
    }
    rows.push(vec![crate::menu_ui::close(locale, "topup:close")]);
    (
        text,
        InlineKeyboardMarkup {
            inline_keyboard: rows,
        },
    )
}

#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn plan_topup_command(
    chat_id: ChatId,
    message_id: MessageId,
    message_text: &str,
    bot_name: &str,
    locale: Locale,
    chat_type: &str,
    billing_available: bool,
    lightning_available: bool,
) -> Option<TelegramAction> {
    let parsed = parse_command(message_text, bot_name);
    if parsed.command != "/topup" {
        return None;
    }
    // "/topup 300" opens on that amount; anything else opens on the default.
    let credits = parse_topup_amount(&parsed.message_text)
        .filter(|credits| (MIN_TOPUP_CREDITS..=MAX_TOPUP_CREDITS).contains(credits))
        .unwrap_or(DEFAULT_TOPUP_CREDITS);
    let (text, keyboard) = if !billing_available {
        (
            crate::billing_commands::billing_unavailable(locale).to_owned(),
            None,
        )
    } else if chat_type != "private" {
        let username = bot_name.trim().trim_start_matches('@');
        (
            match (locale, username.is_empty()) {
                (Locale::Es, false) => format!("La recarga va por privado: abrime en @{username}"),
                (Locale::En, false) => format!("Top-ups happen in private: open @{username}"),
                (Locale::Es, true) => {
                    "La recarga va por privado: escribime por mensaje directo".to_owned()
                }
                (Locale::En, true) => {
                    "Top-ups happen in private: send me a direct message".to_owned()
                }
            },
            (!username.is_empty()).then(|| InlineKeyboardMarkup {
                inline_keyboard: vec![vec![InlineKeyboardButton {
                    text: match locale {
                        Locale::Es => "Abrir chat privado".to_owned(),
                        Locale::En => "Open private chat".to_owned(),
                    },
                    url: Some(format!("https://t.me/{username}")),
                    callback_data: None,
                    copy_text: None,
                }]],
            }),
        )
    } else {
        let (text, keyboard) = topup_menu(locale, credits, lightning_available);
        (text, Some(keyboard))
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
            crate::billing_commands::billing_unavailable(context.locale),
        ));
    }
    let Some(user_id) = context.user_id else {
        return BalanceCommandPlan::Reply(reply(
            context.chat_id,
            context.message_id,
            match context.locale {
                Locale::Es => "No pude identificar tu usuario para ver los saldos",
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
    let user = display_credit_units(CreditUnits::new(user_balance));
    match (chat_balance, locale) {
        (None, Locale::Es) => {
            format!("Saldo de IA: {user} créditos\n\nCargá más con /topup")
        }
        (None, Locale::En) => {
            format!("AI balance: {user} credits\n\nAdd more with /topup")
        }
        (Some(chat), Locale::Es) => {
            let chat = display_credit_units(CreditUnits::new(chat));
            format!(
                "Saldos de IA\n\nTuyo: {user} créditos\nDel grupo: {chat} créditos\n\nPrimero uso tu saldo y, si no alcanza, el del grupo.\n\nCargar: /topup por privado\nPasar al grupo: /transfer <monto>"
            )
        }
        (Some(chat), Locale::En) => {
            let chat = display_credit_units(CreditUnits::new(chat));
            format!(
                "AI balances\n\nYours: {user} credits\nGroup: {chat} credits\n\nI use your balance first, then the group's.\n\nAdd credits: /topup in private\nMove to group: /transfer <amount>"
            )
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopupCallbackPlan {
    Answer(Option<TelegramAction>),
    /// Show the menu again on another amount, in place.
    Menu(i64),
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
    let credits = whole_credits(pack.credits_awarded);
    let (title, description, label) = match locale {
        Locale::Es => (
            format!("{credits} créditos de IA"),
            format!("Recarga de {credits} créditos para mensajes de IA"),
            format!("{credits} créditos de IA"),
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
        return alert(crate::billing_commands::billing_unavailable(locale));
    }
    if chat_type != "private" {
        return alert(match locale {
            Locale::Es => "Cargá por privado, maestro",
            Locale::En => "Open this in a private chat",
        });
    }
    if let Some(credits) = data
        .strip_prefix("topup:amt:")
        .and_then(|credits| credits.parse().ok())
        .filter(|credits| (MIN_TOPUP_CREDITS..=MAX_TOPUP_CREDITS).contains(credits))
    {
        return TopupCallbackPlan::Menu(credits);
    }
    let pack = data
        .split_once(':')
        .filter(|(prefix, _)| *prefix == "topup")
        .and_then(|(_, pack_id)| default_billing_pack(pack_id));
    let Some(pack) = pack else {
        return alert(match locale {
            Locale::Es => "Ese pack es fruta, elegí otro",
            Locale::En => "That credit pack is invalid, choose another one",
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
                    Locale::Es => "Listo, te dejé la factura",
                    Locale::En => "Invoice ready",
                }
                .to_owned(),
            ),
            false,
        ),
        failure_answer: callback_answer(
            callback_id,
            Some(
                match locale {
                    Locale::Es => "No pude armar la factura. Probá de nuevo",
                    Locale::En => "I could not create the invoice. Try again",
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
    let credits = display_credit_units(CreditUnits::new(credits_awarded));
    let balance = display_credit_units(CreditUnits::new(user_balance));
    match (inserted, locale) {
        (true, Locale::Es) => {
            format!("Recarga acreditada: +{credits} créditos\nSaldo personal: {balance} créditos")
        }
        (true, Locale::En) => {
            format!("Top-up complete: +{credits} credits\nPersonal balance: {balance} credits")
        }
        (false, Locale::Es) => {
            format!("Ese pago ya estaba acreditado\nSaldo personal: {balance} créditos")
        }
        (false, Locale::En) => {
            format!("This payment was already credited\nPersonal balance: {balance} credits")
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
                error_message: Some(
                    crate::billing_commands::billing_unavailable(locale).to_owned(),
                ),
            })
        }
        PreCheckoutDecision::InvalidUser { query_id } => Some(TelegramAction::AnswerPreCheckout {
            query_id,
            ok: false,
            error_message: Some(match locale {
                Locale::Es => "No pude identificar tu usuario para cobrarte".to_owned(),
                Locale::En => "I could not identify your user for this payment".to_owned(),
            }),
        }),
        PreCheckoutDecision::InvalidPayment { query_id } => {
            Some(TelegramAction::AnswerPreCheckout {
                query_id,
                ok: false,
                error_message: Some(match locale {
                    Locale::Es => "Ese pago vino raro y no te lo pude validar".to_owned(),
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
        parse_topup_amount, parse_topup_payload, payment_record, plan_balance_command,
        plan_pre_checkout, plan_topup_callback, plan_topup_command, successful_payment_reply,
        topup_menu, topup_menu_title,
    };
    use crate::locale::Locale;
    use crate::telegram_actions::{InlineKeyboardButton, LabeledPrice, TelegramAction};
    use crate::telegram_input::{ChatId, MessageId};

    fn sent(action: Option<TelegramAction>) -> Option<crate::telegram_actions::SendMessage> {
        match action {
            Some(TelegramAction::SendMessage(message)) => Some(message),
            _ => None,
        }
    }

    fn balance_message(plan: BalanceCommandPlan) -> Option<crate::telegram_actions::SendMessage> {
        match plan {
            BalanceCommandPlan::Reply(action) => sent(Some(action)),
            _ => None,
        }
    }

    #[test]
    fn what_a_pack_nets_covers_its_credits_ai_cost_plus_the_markup() {
        use crate::provider_pricing::{
            AI_MARKUP_PERCENT, CREDIT_UNIT_USD_MICROS, STAR_PAYOUT_USD_MICROS,
        };
        assert_eq!(CREDIT_UNIT_USD_MICROS, 50);
        for pack in super::billing_packs() {
            let payout = i128::from(pack.xtr_amount) * STAR_PAYOUT_USD_MICROS;
            let ai_cost = i128::from(pack.credits_awarded) * CREDIT_UNIT_USD_MICROS;
            assert_eq!(payout * 100, ai_cost * (100 + AI_MARKUP_PERCENT));
        }
    }

    fn pack() -> BillingPackTerms {
        BillingPackTerms {
            id: "p50".to_owned(),
            xtr_amount: 25,
            credits_awarded: 5_000,
        }
    }

    #[test]
    fn any_amount_in_range_is_a_pack_at_two_credits_per_star() {
        assert_eq!(
            default_billing_pack("c300"),
            Some(BillingPackTerms {
                id: "c300".to_owned(),
                xtr_amount: 150,
                credits_awarded: 30_000,
            })
        );
        // Odd amounts round the Stars up.
        assert_eq!(
            default_billing_pack("c11").map(|pack| pack.xtr_amount),
            Some(6)
        );
        assert_eq!(
            default_billing_pack("c100000").map(|pack| pack.xtr_amount),
            Some(50_000)
        );
        assert_eq!(
            default_billing_pack("c10").map(|pack| pack.xtr_amount),
            Some(5)
        );
        for invalid in ["c9", "c100001", "c050", "c", "c-20", "cabc", "c+20"] {
            assert_eq!(default_billing_pack(invalid), None, "{invalid}");
        }
        for (text, expected) in [
            ("300", Some(300)),
            (" 1.500 ", Some(1_500)),
            ("1,500", Some(1_500)),
            ("10 000", Some(10_000)),
            ("", None),
            ("-5", None),
            ("3.5k", None),
            ("💰", None),
            ("99999999999999999999", None),
        ] {
            assert_eq!(parse_topup_amount(text), expected, "{text}");
        }
    }

    #[test]
    fn amount_callbacks_redraw_the_menu_only_in_range() {
        let plan = |data: &str| {
            plan_topup_callback(
                Some("cb"),
                data,
                ChatId(42),
                "private",
                Some(42),
                true,
                Locale::En,
            )
        };
        assert_eq!(plan("topup:amt:300"), TopupCallbackPlan::Menu(300));
        assert_eq!(plan("topup:amt:10"), TopupCallbackPlan::Menu(10));
        assert_eq!(plan("topup:amt:100000"), TopupCallbackPlan::Menu(100_000));
        for invalid in ["topup:amt:9", "topup:amt:100001", "topup:amt:x"] {
            assert!(
                matches!(plan(invalid), TopupCallbackPlan::Answer(Some(_))),
                "{invalid}"
            );
        }
        assert!(matches!(plan("topup:c300"), TopupCallbackPlan::Invoice(_)));
    }

    #[test]
    fn menu_steps_quick_picks_and_payment_buttons_follow_the_amount() {
        let rows =
            |credits, lightning| topup_menu(Locale::En, credits, lightning).1.inline_keyboard;
        let callbacks = |row: &Vec<InlineKeyboardButton>| {
            row.iter()
                .map(|button| button.callback_data.clone().unwrap_or_default())
                .collect::<Vec<_>>()
        };
        for (credits, fewer, more) in [
            (10, 10, 20),
            (100, 90, 150),
            (1_000, 950, 1_500),
            (10_000, 9_500, 15_000),
            (100_000, 95_000, 100_000),
        ] {
            assert_eq!(
                callbacks(&rows(credits, false)[0]),
                [
                    format!("topup:amt:{fewer}"),
                    format!("topup:amt:{credits}"),
                    format!("topup:amt:{more}"),
                ],
                "{credits}"
            );
        }
        let menu = rows(500, true);
        assert_eq!(
            menu[1]
                .iter()
                .map(|button| button.text.as_str())
                .collect::<Vec<_>>(),
            ["50", "100", "✓ 500", "1,000"]
        );
        assert_eq!(menu[2][0].text, "💳 Telegram  250 ⭐");
        assert_eq!(menu[3][0].text, "⚡ Lightning  US$3.29");
        assert_eq!(menu[3][0].callback_data.as_deref(), Some("topup:ln:c500"));
        assert_eq!(menu[4][0].callback_data.as_deref(), Some("topup:close"));
        // Out of range clamps to the nearest end.
        assert_eq!(rows(1, false)[0][1].text, "10 credits");
        assert_eq!(rows(1_000_000, false)[0][1].text, "100,000 credits");
        assert_eq!(topup_menu_title(Locale::Es), "Cargar créditos");
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
    fn topup_command_plans_private_catalog_group_redirect_and_unavailable_reply()
    -> Result<(), String> {
        let private = plan_topup_command(
            ChatId(42),
            MessageId(7),
            "/topup@mybot",
            "@mybot",
            Locale::En,
            "private",
            true,
            false,
        );
        let private = sent(private).ok_or("private topup message")?;
        assert_eq!(
            private.text,
            "Add credits\n\n100 credits ≈ 400 AI messages\nChange the amount or send me a number"
        );
        assert_eq!(private.reply_to_message_id, Some(MessageId(7)));
        let keyboard =
            private
                .reply_markup
                .unwrap_or(crate::telegram_actions::InlineKeyboardMarkup {
                    inline_keyboard: Vec::new(),
                });
        // Amount, quick picks, Telegram and close: no Lightning row without it.
        assert_eq!(keyboard.inline_keyboard.len(), 4);
        assert_eq!(keyboard.inline_keyboard[2][0].text, "💳 Telegram  50 ⭐");
        assert_eq!(
            keyboard.inline_keyboard[2][0].callback_data.as_deref(),
            Some("topup:c100")
        );

        // A typed amount opens the menu on it; anything else on the default.
        for (text, expected) in [
            ("/topup 300", "300 credits ≈ 1,200 AI messages"),
            ("/topup 1.500", "1,500 credits ≈ 6,000 AI messages"),
            ("/topup 100000", "100,000 credits ≈ 400,000 AI messages"),
            ("/topup 9", "100 credits"),
            ("/topup 100001", "100 credits"),
            ("/topup lots", "100 credits"),
        ] {
            let message = sent(plan_topup_command(
                ChatId(42),
                MessageId(7),
                text,
                "@mybot",
                Locale::En,
                "private",
                true,
                false,
            ))
            .ok_or("typed topup")?;
            assert!(
                message
                    .text
                    .starts_with(&format!("Add credits\n\n{expected}")),
                "{text}"
            );
        }

        for (chat_type, available, bot_name, locale, expected) in [
            (
                "group",
                true,
                "@mybot",
                Locale::Es,
                "La recarga va por privado: abrime en @mybot",
            ),
            (
                "group",
                true,
                "",
                Locale::En,
                "Top-ups happen in private: send me a direct message",
            ),
            (
                "private",
                false,
                "@mybot",
                Locale::En,
                "AI credits are unavailable right now. Try again later or tell the admin",
            ),
        ] {
            let message = sent(plan_topup_command(
                ChatId(42),
                MessageId(7),
                "/topup",
                bot_name,
                locale,
                chat_type,
                available,
                false,
            ))
            .ok_or("topup message")?;
            assert_eq!(message.text, expected);
            let url = message
                .reply_markup
                .and_then(|markup| markup.inline_keyboard.into_iter().flatten().next())
                .and_then(|button| button.url);
            let expected_url = (chat_type == "group" && !bot_name.is_empty())
                .then(|| "https://t.me/mybot".to_owned());
            assert_eq!(url, expected_url);
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
                false,
            ),
            None
        );
        Ok(())
    }

    #[test]
    fn balance_command_plans_external_loads_and_early_replies() -> Result<(), String> {
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
                "AI credits are unavailable right now. Try again later or tell the admin",
            ),
            (
                None,
                true,
                Locale::Es,
                "No pude identificar tu usuario para ver los saldos",
            ),
        ] {
            let plan = plan_balance_command(
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
            );
            let message = balance_message(plan).ok_or("balance reply")?;
            assert_eq!(message.text, expected);
            assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
        }
        Ok(())
    }

    #[test]
    fn balance_replies_match_private_and_group_credit_formatting() {
        assert_eq!(
            balance_reply(4_200, None, Locale::Es),
            "Saldo de IA: 42.00 créditos\n\nCargá más con /topup"
        );
        assert_eq!(
            balance_reply(4_200, None, Locale::En),
            "AI balance: 42.00 credits\n\nAdd more with /topup"
        );
        assert_eq!(
            balance_reply(3_000, Some(12_000), Locale::Es),
            "Saldos de IA\n\nTuyo: 30.00 créditos\nDel grupo: 120.00 créditos\n\nPrimero uso tu saldo y, si no alcanza, el del grupo.\n\nCargar: /topup por privado\nPasar al grupo: /transfer <monto>"
        );
        assert_eq!(
            balance_reply(3_000, Some(12_000), Locale::En),
            "AI balances\n\nYours: 30.00 credits\nGroup: 120.00 credits\n\nI use your balance first, then the group's.\n\nAdd credits: /topup in private\nMove to group: /transfer <amount>"
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
                    title: "50 AI credit pack".to_owned(),
                    description: "Add 50 credits for AI messages".to_owned(),
                    payload: "topup:p50:42:en".to_owned(),
                    currency: "XTR".to_owned(),
                    prices: vec![LabeledPrice {
                        label: "50 AI credits".to_owned(),
                        amount: 25,
                    }],
                },
                success_answer: Some(TelegramAction::AnswerCallback {
                    callback_id: "callback-1".to_owned(),
                    text: Some("Invoice ready".to_owned()),
                    show_alert: false,
                }),
                failure_answer: Some(TelegramAction::AnswerCallback {
                    callback_id: "callback-1".to_owned(),
                    text: Some("I could not create the invoice. Try again".to_owned()),
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
                Some(
                    "Los créditos de IA no están disponibles en este momento. Probá más tarde o avisale al admin",
                ),
                true,
            ),
            (
                "topup:p50",
                "group",
                Some(42),
                true,
                Locale::En,
                Some("Open this in a private chat"),
                true,
            ),
            (
                "topup:missing",
                "private",
                Some(42),
                true,
                Locale::Es,
                Some("Ese pack es fruta, elegí otro"),
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
                "Los créditos de IA no están disponibles en este momento. Probá más tarde o avisale al admin",
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
                "Ese pago vino raro y no te lo pude validar",
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
            "Recarga acreditada: +50.00 créditos\nSaldo personal: 53.00 créditos"
        );
        assert_eq!(
            successful_payment_reply(5_000, 5_300, true, Locale::En),
            "Top-up complete: +50.00 credits\nPersonal balance: 53.00 credits"
        );
        assert_eq!(
            successful_payment_reply(5_000, 5_300, false, Locale::Es),
            "Ese pago ya estaba acreditado\nSaldo personal: 53.00 créditos"
        );
        assert_eq!(
            successful_payment_reply(5_000, 5_300, false, Locale::En),
            "This payment was already credited\nPersonal balance: 53.00 credits"
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

    #[test]
    fn spanish_and_english_topup_variants_are_complete() -> Result<(), String> {
        let private = sent(plan_topup_command(
            ChatId(42),
            MessageId(7),
            "/topup",
            "@mybot",
            Locale::Es,
            "private",
            true,
            false,
        ))
        .ok_or("spanish catalog")?;
        assert_eq!(
            private.text,
            "Cargar créditos\n\n100 créditos ≈ 400 mensajes de IA\nCambiá el monto o mandame el número"
        );
        let amount = private
            .reply_markup
            .and_then(|markup| markup.inline_keyboard.into_iter().next())
            .and_then(|row| row.into_iter().nth(1))
            .map(|button| button.text);
        assert_eq!(amount.as_deref(), Some("100 créditos"));

        let group = sent(plan_topup_command(
            ChatId(-42),
            MessageId(7),
            "/topup",
            "@mybot",
            Locale::En,
            "supergroup",
            true,
            false,
        ))
        .ok_or("english redirect")?;
        assert_eq!(group.text, "Top-ups happen in private: open @mybot");
        let button = group
            .reply_markup
            .and_then(|markup| markup.inline_keyboard.into_iter().flatten().next())
            .map(|button| (button.text, button.url));
        assert_eq!(
            button,
            Some((
                "Open private chat".to_owned(),
                Some("https://t.me/mybot".to_owned())
            ))
        );

        let anonymous = sent(plan_topup_command(
            ChatId(-42),
            MessageId(7),
            "/topup",
            " @ ",
            Locale::Es,
            "group",
            true,
            false,
        ))
        .ok_or("spanish redirect")?;
        assert_eq!(
            anonymous.text,
            "La recarga va por privado: escribime por mensaje directo"
        );
        assert!(anonymous.reply_markup.is_none());
        assert_eq!(
            sent(Some(TelegramAction::DeleteMessage {
                chat_id: ChatId(1),
                message_id: MessageId(1),
            })),
            None
        );

        let balance = plan_balance_command(
            "/balance",
            "@mybot",
            BalanceCommandContext {
                chat_id: ChatId(42),
                message_id: MessageId(7),
                user_id: None,
                locale: Locale::En,
                is_group: true,
                billing_available: true,
            },
        );
        assert_eq!(
            balance_message(balance)
                .map(|message| message.text)
                .as_deref(),
            Some("I could not identify the user or chat to load the balances")
        );
        assert_eq!(balance_message(BalanceCommandPlan::NotHandled), None);
        Ok(())
    }

    #[test]
    fn spanish_invoices_and_guard_answers_are_localized() {
        let plan = plan_topup_callback(
            Some("cb"),
            "topup:p100",
            ChatId(42),
            "private",
            Some(42),
            true,
            Locale::Es,
        );
        assert_eq!(
            plan,
            TopupCallbackPlan::Invoice(Box::new(super::TopupInvoicePlan {
                invoice: TelegramAction::SendInvoice {
                    chat_id: ChatId(42),
                    title: "100 créditos de IA".to_owned(),
                    description: "Recarga de 100 créditos para mensajes de IA".to_owned(),
                    payload: "topup:p100:42:es".to_owned(),
                    currency: "XTR".to_owned(),
                    prices: vec![LabeledPrice {
                        label: "100 créditos de IA".to_owned(),
                        amount: 50,
                    }],
                },
                success_answer: Some(TelegramAction::AnswerCallback {
                    callback_id: "cb".to_owned(),
                    text: Some("Listo, te dejé la factura".to_owned()),
                    show_alert: false,
                }),
                failure_answer: Some(TelegramAction::AnswerCallback {
                    callback_id: "cb".to_owned(),
                    text: Some("No pude armar la factura. Probá de nuevo".to_owned()),
                    show_alert: true,
                }),
            }))
        );
        for (data, chat_type, locale, expected) in [
            (
                "topup:p50",
                "group",
                Locale::Es,
                "Cargá por privado, maestro",
            ),
            (
                "other:p50",
                "private",
                Locale::En,
                "That credit pack is invalid, choose another one",
            ),
        ] {
            assert_eq!(
                plan_topup_callback(
                    Some("cb"),
                    data,
                    ChatId(42),
                    chat_type,
                    Some(42),
                    true,
                    locale
                ),
                TopupCallbackPlan::Answer(Some(TelegramAction::AnswerCallback {
                    callback_id: "cb".to_owned(),
                    text: Some(expected.to_owned()),
                    show_alert: true,
                }))
            );
        }
    }

    #[test]
    fn pre_checkout_rejections_are_localized() {
        assert_eq!(
            plan_pre_checkout(
                &json!({"id": "q1", "from": {"id": "not-a-number"}, "invoice_payload": "topup:p50:42:es"}),
                true,
                Locale::Es,
            ),
            Ok(Some(TelegramAction::AnswerPreCheckout {
                query_id: "q1".to_owned(),
                ok: false,
                error_message: Some("No pude identificar tu usuario para cobrarte".to_owned()),
            }))
        );
        assert_eq!(
            plan_pre_checkout(
                &json!({"id": "q2", "from": {"id": 42}, "invoice_payload": "topup:p50:42:en", "currency": "USD", "total_amount": 25}),
                true,
                Locale::En,
            ),
            Ok(Some(TelegramAction::AnswerPreCheckout {
                query_id: "q2".to_owned(),
                ok: false,
                error_message: Some("I could not validate this payment".to_owned()),
            }))
        );
        assert_eq!(super::whole_credits(150), "1.50");
        assert_eq!(super::whole_credits(5_000), "50");
    }
}
