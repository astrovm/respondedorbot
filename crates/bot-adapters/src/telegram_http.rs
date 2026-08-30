//! Blocking Telegram Bot API transport for non-multipart requests.

use std::io::Read;
use std::time::Duration;

use reqwest::Method;
use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::Value;
use thiserror::Error;

const TELEGRAM_API_BASE: &str = "https://api.telegram.org";
const MAX_RESPONSE_BYTES: u64 = 1_048_576;

#[derive(Clone, PartialEq)]
pub struct TelegramRequest {
    pub token: String,
    pub endpoint: String,
    pub method: Method,
    pub params: Option<Value>,
    pub json_payload: Option<Value>,
    pub timeout: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status_code: u16,
    pub body: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportFailureKind {
    Timeout,
    Connection,
    Request,
    ResponseTooLarge,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TelegramHttpError {
    #[error("Telegram HTTP method is invalid")]
    InvalidMethod,
    #[error("Telegram timeout must be positive")]
    InvalidTimeout,
    #[error("Telegram HTTP payload must be a JSON object")]
    InvalidPayload,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TelegramHttpOutcome {
    Response { status_code: u16, body: String },
    TransportError { kind: TransportFailureKind },
}

pub trait TelegramTransport {
    fn send(&self, request: &TelegramRequest) -> Result<HttpResponse, TransportFailureKind>;
}

pub struct ReqwestTelegramTransport {
    client: Client,
}

impl ReqwestTelegramTransport {
    pub fn new() -> Result<Self, TransportFailureKind> {
        Client::builder()
            .build()
            .map(|client| Self { client })
            .map_err(|_| TransportFailureKind::Request)
    }
}

impl TelegramTransport for ReqwestTelegramTransport {
    fn send(&self, request: &TelegramRequest) -> Result<HttpResponse, TransportFailureKind> {
        let url = format!(
            "{TELEGRAM_API_BASE}/bot{}/{}",
            request.token, request.endpoint
        );
        let mut builder = self
            .client
            .request(request.method.clone(), url)
            .timeout(request.timeout);
        if let Some(params) = &request.params {
            builder = builder.query(params);
        }
        if let Some(payload) = &request.json_payload {
            builder = builder.json(payload);
        }
        let response = builder.send().map_err(classify_error)?;
        let status_code = response.status().as_u16();
        let mut body = Vec::new();
        response
            .take(MAX_RESPONSE_BYTES + 1)
            .read_to_end(&mut body)
            .map_err(|_| TransportFailureKind::Request)?;
        if body.len() as u64 > MAX_RESPONSE_BYTES {
            return Err(TransportFailureKind::ResponseTooLarge);
        }
        Ok(HttpResponse {
            status_code,
            body: String::from_utf8_lossy(&body).into_owned(),
        })
    }
}

fn classify_error(error: reqwest::Error) -> TransportFailureKind {
    if error.is_timeout() {
        TransportFailureKind::Timeout
    } else if error.is_connect() {
        TransportFailureKind::Connection
    } else {
        TransportFailureKind::Request
    }
}

fn validate_payload(payload: Option<Value>) -> Result<Option<Value>, TelegramHttpError> {
    match payload {
        Some(Value::Object(object)) => Ok(Some(Value::Object(object))),
        Some(_) => Err(TelegramHttpError::InvalidPayload),
        None => Ok(None),
    }
}

pub fn request_with<T: TelegramTransport>(
    transport: &T,
    token: &str,
    endpoint: &str,
    method: &str,
    params: Option<Value>,
    json_payload: Option<Value>,
    timeout_seconds: u64,
) -> Result<TelegramHttpOutcome, TelegramHttpError> {
    if timeout_seconds == 0 {
        return Err(TelegramHttpError::InvalidTimeout);
    }
    let method = Method::from_bytes(method.to_uppercase().as_bytes())
        .map_err(|_| TelegramHttpError::InvalidMethod)?;
    let request = TelegramRequest {
        token: token.to_owned(),
        endpoint: endpoint.to_owned(),
        method,
        params: validate_payload(params)?,
        json_payload: validate_payload(json_payload)?,
        timeout: Duration::from_secs(timeout_seconds),
    };
    match transport.send(&request) {
        Ok(response) => Ok(response_outcome(response.status_code, response.body)),
        Err(kind) => Ok(TelegramHttpOutcome::TransportError { kind }),
    }
}

#[must_use]
pub fn response_outcome(status_code: u16, body: String) -> TelegramHttpOutcome {
    TelegramHttpOutcome::Response { status_code, body }
}

pub fn request(
    token: &str,
    endpoint: &str,
    method: &str,
    params: Option<Value>,
    json_payload: Option<Value>,
    timeout_seconds: u64,
) -> Result<TelegramHttpOutcome, TelegramHttpError> {
    match ReqwestTelegramTransport::new() {
        Ok(transport) => request_with(
            &transport,
            token,
            endpoint,
            method,
            params,
            json_payload,
            timeout_seconds,
        ),
        Err(kind) => Ok(TelegramHttpOutcome::TransportError { kind }),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use reqwest::Method;
    use serde_json::json;

    use super::{
        HttpResponse, TelegramHttpError, TelegramHttpOutcome, TelegramRequest, TelegramTransport,
        TransportFailureKind, request_with, response_outcome,
    };

    struct FakeTransport {
        response: RefCell<Option<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<TelegramRequest>>,
    }

    impl TelegramTransport for FakeTransport {
        fn send(&self, request: &TelegramRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            self.response
                .borrow_mut()
                .take()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    fn transport(response: Result<HttpResponse, TransportFailureKind>) -> FakeTransport {
        FakeTransport {
            response: RefCell::new(Some(response)),
            requests: RefCell::new(Vec::new()),
        }
    }

    #[test]
    fn request_preserves_method_payload_timeout_and_identity() {
        let transport = transport(Ok(HttpResponse {
            status_code: 200,
            body: r#"{"ok":true}"#.to_owned(),
        }));
        let actual = request_with(
            &transport,
            "synthetic-token",
            "sendMessage",
            "post",
            None,
            Some(json!({"chat_id": "42", "text": "hola"})),
            5,
        );
        assert_eq!(
            actual,
            Ok(TelegramHttpOutcome::Response {
                status_code: 200,
                body: r#"{"ok":true}"#.to_owned(),
            })
        );
        let requests = transport.requests.borrow();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].token, "synthetic-token");
        assert_eq!(requests[0].endpoint, "sendMessage");
        assert_eq!(requests[0].method, Method::POST);
        assert_eq!(requests[0].params, None);
        assert_eq!(
            requests[0].json_payload,
            Some(json!({"chat_id": "42", "text": "hola"}))
        );
        assert_eq!(requests[0].timeout.as_secs(), 5);
    }

    #[test]
    fn transport_failures_are_serializable_outcomes() {
        for kind in [
            TransportFailureKind::Timeout,
            TransportFailureKind::Connection,
            TransportFailureKind::Request,
            TransportFailureKind::ResponseTooLarge,
        ] {
            let transport = transport(Err(kind));
            assert_eq!(
                request_with(&transport, "token", "getMe", "GET", None, None, 5),
                Ok(TelegramHttpOutcome::TransportError { kind })
            );
        }
    }

    #[test]
    fn invalid_method_timeout_and_payload_are_rejected_before_send() {
        let transport = transport(Err(TransportFailureKind::Request));
        assert_eq!(
            request_with(&transport, "token", "endpoint", "bad method", None, None, 5),
            Err(TelegramHttpError::InvalidMethod)
        );
        assert_eq!(
            request_with(&transport, "token", "endpoint", "GET", None, None, 0),
            Err(TelegramHttpError::InvalidTimeout)
        );
        assert_eq!(
            request_with(
                &transport,
                "token",
                "endpoint",
                "GET",
                Some(json!([])),
                None,
                5,
            ),
            Err(TelegramHttpError::InvalidPayload)
        );
        assert!(transport.requests.borrow().is_empty());
    }

    #[test]
    fn response_classification_preserves_status_and_body_verbatim() {
        assert_eq!(
            response_outcome(429, "  body  ".to_owned()),
            TelegramHttpOutcome::Response {
                status_code: 429,
                body: "  body  ".to_owned(),
            }
        );
    }
}
