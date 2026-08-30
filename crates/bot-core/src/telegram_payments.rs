//! Telegram Stars pre-checkout validation.

use serde::Serialize;
use serde_json::Value;
use thiserror::Error;

use crate::telegram_input::{python_string, python_truthy};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BillingPackTerms {
    pub id: String,
    pub xtr_amount: i64,
    pub credits_awarded: i64,
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
        BillingPackTerms, PaymentValidationError, PreCheckoutDecision, SuccessfulPaymentDecision,
        evaluate_pre_checkout, evaluate_successful_payment, parse_topup_payload,
    };

    fn pack() -> BillingPackTerms {
        BillingPackTerms {
            id: "p50".to_owned(),
            xtr_amount: 25,
            credits_awarded: 5_000,
        }
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
