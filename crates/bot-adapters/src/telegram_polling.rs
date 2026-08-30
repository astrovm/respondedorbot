//! Telegram `getUpdates` request and response handling.
//!
//! This module owns the untrusted JSON boundary. Callers receive an update id
//! and one of the three update kinds supported by the bot runtime.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::telegram_http::{
    TelegramHttpError, TelegramHttpOutcome, TelegramTransport, TransportFailureKind, request_with,
};

pub const DEFAULT_LONG_POLL_SECONDS: u64 = 30;
const HTTP_TIMEOUT_MARGIN_SECONDS: u64 = 5;

#[derive(Debug, Clone, PartialEq)]
pub struct IncomingUpdate {
    pub update_id: i64,
    pub event: IncomingEvent,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IncomingEvent {
    Message(Map<String, Value>),
    CallbackQuery(Map<String, Value>),
    PreCheckoutQuery(Map<String, Value>),
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PollFailure {
    Transport { failure: TransportFailureKind },
    Http { status_code: u16 },
    Conflict,
    RateLimited { retry_after_seconds: Option<u64> },
    Api { error_code: Option<i64> },
}

#[derive(Debug, Clone, PartialEq)]
pub enum PollOutcome {
    Updates(Vec<IncomingUpdate>),
    Retry(PollFailure),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PollingError {
    #[error("Telegram polling timeout must be positive")]
    InvalidTimeout,
    #[error("Telegram polling response was malformed")]
    InvalidResponse,
    #[error(transparent)]
    Http(#[from] TelegramHttpError),
}

#[derive(Debug, Deserialize)]
struct ApiEnvelope {
    ok: bool,
    #[serde(default)]
    result: Option<Vec<Value>>,
    #[serde(default)]
    error_code: Option<i64>,
    #[serde(default)]
    parameters: Option<ApiErrorParameters>,
}

#[derive(Debug, Deserialize)]
struct ApiErrorParameters {
    #[serde(default)]
    retry_after: Option<u64>,
}

#[must_use]
pub fn next_offset(updates: &[IncomingUpdate], current: Option<i64>) -> Option<i64> {
    updates
        .iter()
        .map(|update| update.update_id)
        .max()
        .and_then(|update_id| update_id.checked_add(1))
        .or(current)
}

fn parse_event(object: &mut Map<String, Value>) -> Result<IncomingEvent, PollingError> {
    let mut supported = Vec::new();
    for field in ["message", "callback_query", "pre_checkout_query"] {
        if object.contains_key(field) {
            supported.push(field);
        }
    }
    if supported.len() > 1 {
        return Err(PollingError::InvalidResponse);
    }
    let Some(field) = supported.first() else {
        return Ok(IncomingEvent::Unsupported);
    };
    let payload = object
        .remove(*field)
        .and_then(|value| value.as_object().cloned())
        .ok_or(PollingError::InvalidResponse)?;
    Ok(match *field {
        "message" => IncomingEvent::Message(payload),
        "callback_query" => IncomingEvent::CallbackQuery(payload),
        "pre_checkout_query" => IncomingEvent::PreCheckoutQuery(payload),
        _ => IncomingEvent::Unsupported,
    })
}

fn parse_update(value: Value) -> Result<IncomingUpdate, PollingError> {
    let mut object = value
        .as_object()
        .cloned()
        .ok_or(PollingError::InvalidResponse)?;
    let update_id = object
        .remove("update_id")
        .and_then(|value| value.as_i64())
        .ok_or(PollingError::InvalidResponse)?;
    Ok(IncomingUpdate {
        update_id,
        event: parse_event(&mut object)?,
    })
}

pub fn parse_response(status_code: u16, body: &str) -> Result<PollOutcome, PollingError> {
    if !(200..300).contains(&status_code) {
        return Ok(PollOutcome::Retry(PollFailure::Http { status_code }));
    }
    let envelope: ApiEnvelope =
        serde_json::from_str(body).map_err(|_| PollingError::InvalidResponse)?;
    if !envelope.ok {
        let failure = match envelope.error_code {
            Some(409) => PollFailure::Conflict,
            Some(429) => PollFailure::RateLimited {
                retry_after_seconds: envelope.parameters.and_then(|value| value.retry_after),
            },
            error_code => PollFailure::Api { error_code },
        };
        return Ok(PollOutcome::Retry(failure));
    }
    let result = envelope.result.ok_or(PollingError::InvalidResponse)?;
    let updates = result
        .into_iter()
        .map(parse_update)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PollOutcome::Updates(updates))
}

pub fn poll_once_with<T: TelegramTransport>(
    transport: &T,
    token: &str,
    offset: Option<i64>,
    long_poll_seconds: u64,
) -> Result<PollOutcome, PollingError> {
    if long_poll_seconds == 0 {
        return Err(PollingError::InvalidTimeout);
    }
    let request_timeout = long_poll_seconds
        .checked_add(HTTP_TIMEOUT_MARGIN_SECONDS)
        .ok_or(PollingError::InvalidTimeout)?;
    let mut params = json!({
        "timeout": long_poll_seconds,
        "allowed_updates": ["message", "callback_query", "pre_checkout_query"]
    });
    if let Some(offset) = offset {
        params["offset"] = json!(offset);
    }
    match request_with(
        transport,
        token,
        "getUpdates",
        "GET",
        Some(params),
        None,
        request_timeout,
    )? {
        TelegramHttpOutcome::Response { status_code, body } => parse_response(status_code, &body),
        TelegramHttpOutcome::TransportError { kind } => {
            Ok(PollOutcome::Retry(PollFailure::Transport { failure: kind }))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use reqwest::Method;
    use serde_json::json;

    use super::{
        IncomingEvent, PollFailure, PollOutcome, PollingError, next_offset, parse_response,
        poll_once_with,
    };
    use crate::telegram_http::{
        HttpResponse, TelegramRequest, TelegramTransport, TransportFailureKind,
    };

    struct FakeTransport {
        result: RefCell<Option<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<TelegramRequest>>,
    }

    impl TelegramTransport for FakeTransport {
        fn send(&self, request: &TelegramRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            self.result
                .borrow_mut()
                .take()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    fn transport(result: Result<HttpResponse, TransportFailureKind>) -> FakeTransport {
        FakeTransport {
            result: RefCell::new(Some(result)),
            requests: RefCell::new(Vec::new()),
        }
    }

    #[test]
    fn poll_request_has_stable_offset_allowed_updates_and_timeout_margin() {
        let transport = transport(Ok(HttpResponse {
            status_code: 200,
            body: r#"{"ok":true,"result":[]}"#.to_owned(),
        }));
        assert_eq!(
            poll_once_with(&transport, "synthetic-token", Some(42), 30),
            Ok(PollOutcome::Updates(Vec::new()))
        );
        let requests = transport.requests.borrow();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].endpoint, "getUpdates");
        assert_eq!(requests[0].method, Method::GET);
        assert_eq!(requests[0].timeout.as_secs(), 35);
        assert_eq!(
            requests[0].params,
            Some(json!({
                "offset": 42,
                "timeout": 30,
                "allowed_updates": ["message", "callback_query", "pre_checkout_query"]
            }))
        );
    }

    #[test]
    fn response_decodes_supported_and_unsupported_updates() {
        let actual = parse_response(
            200,
            r#"{"ok":true,"result":[
                {"update_id":10,"message":{"message_id":1,"text":"hola"}},
                {"update_id":11,"callback_query":{"id":"callback"}},
                {"update_id":12,"pre_checkout_query":{"id":"checkout"}},
                {"update_id":13,"edited_message":{"message_id":2}}
            ]}"#,
        );
        assert!(matches!(&actual, Ok(PollOutcome::Updates(_))));
        let updates = match actual {
            Ok(PollOutcome::Updates(updates)) => updates,
            Ok(PollOutcome::Retry(_)) | Err(_) => Vec::new(),
        };
        assert_eq!(updates.len(), 4);
        assert!(matches!(updates[0].event, IncomingEvent::Message(_)));
        assert!(matches!(updates[1].event, IncomingEvent::CallbackQuery(_)));
        assert!(matches!(
            updates[2].event,
            IncomingEvent::PreCheckoutQuery(_)
        ));
        assert_eq!(updates[3].event, IncomingEvent::Unsupported);
        assert_eq!(next_offset(&updates, None), Some(14));
    }

    #[test]
    fn api_failures_preserve_conflict_rate_limit_and_status_information() {
        assert_eq!(
            parse_response(200, r#"{"ok":false,"error_code":409}"#),
            Ok(PollOutcome::Retry(PollFailure::Conflict))
        );
        assert_eq!(
            parse_response(
                200,
                r#"{"ok":false,"error_code":429,"parameters":{"retry_after":7}}"#,
            ),
            Ok(PollOutcome::Retry(PollFailure::RateLimited {
                retry_after_seconds: Some(7),
            }))
        );
        assert_eq!(
            parse_response(503, "upstream unavailable"),
            Ok(PollOutcome::Retry(PollFailure::Http { status_code: 503 }))
        );
    }

    #[test]
    fn malformed_success_payloads_are_rejected() {
        for body in [
            "not-json",
            r#"{"ok":true}"#,
            r#"{"ok":true,"result":{}}"#,
            r#"{"ok":true,"result":[{}]}"#,
            r#"{"ok":true,"result":[{"update_id":1,"message":[]}]}"#,
            r#"{"ok":true,"result":[{"update_id":1,"message":{},"callback_query":{}}]}"#,
        ] {
            assert_eq!(
                parse_response(200, body),
                Err(PollingError::InvalidResponse)
            );
        }
    }

    #[test]
    fn validation_and_transport_failures_do_not_advance_offset() {
        let invalid = transport(Err(TransportFailureKind::Request));
        assert_eq!(
            poll_once_with(&invalid, "token", None, 0),
            Err(PollingError::InvalidTimeout)
        );
        assert!(invalid.requests.borrow().is_empty());

        let failed = transport(Err(TransportFailureKind::Timeout));
        assert_eq!(
            poll_once_with(&failed, "token", None, 30),
            Ok(PollOutcome::Retry(PollFailure::Transport {
                failure: TransportFailureKind::Timeout,
            }))
        );
        assert_eq!(next_offset(&[], Some(90)), Some(90));
    }

    #[test]
    fn next_offset_uses_highest_update_and_handles_overflow() {
        let first = parse_response(
            200,
            r#"{"ok":true,"result":[{"update_id":8},{"update_id":3}]}"#,
        );
        assert!(matches!(&first, Ok(PollOutcome::Updates(_))));
        let updates = match first {
            Ok(PollOutcome::Updates(updates)) => updates,
            Ok(PollOutcome::Retry(_)) | Err(_) => Vec::new(),
        };
        assert_eq!(next_offset(&updates, Some(2)), Some(9));

        let overflow = parse_response(
            200,
            &format!(r#"{{"ok":true,"result":[{{"update_id":{}}}]}}"#, i64::MAX),
        );
        assert!(matches!(&overflow, Ok(PollOutcome::Updates(_))));
        let updates = match overflow {
            Ok(PollOutcome::Updates(updates)) => updates,
            Ok(PollOutcome::Retry(_)) | Err(_) => Vec::new(),
        };
        assert_eq!(next_offset(&updates, Some(5)), Some(5));
    }
}
