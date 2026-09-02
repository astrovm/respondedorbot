//! Blocking Telegram Bot API transport.

use std::io::Cursor;
use std::io::Read;
use std::sync::Arc;
use std::time::Duration;

use reqwest::Method;
use reqwest::blocking::Client;
use reqwest::blocking::multipart::{Form, Part};
use serde::Serialize;
use serde_json::Value;
use thiserror::Error;

const TELEGRAM_API_BASE: &str = "https://api.telegram.org";
const MAX_RESPONSE_BYTES: u64 = 1_048_576;
pub const TELEGRAM_FILE_MAX_BYTES: u64 = 20_000_000;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryHttpResponse {
    pub status_code: u16,
    pub body: Vec<u8>,
}

#[derive(Clone, PartialEq, Eq)]
pub struct TelegramMultipartRequest {
    pub token: String,
    pub endpoint: String,
    pub fields: Vec<(String, String)>,
    pub file_field: String,
    pub file_name: String,
    pub file_bytes: Arc<[u8]>,
    pub content_type: String,
    pub timeout: Duration,
}

#[derive(Clone, PartialEq, Eq)]
pub struct TelegramFileRequest {
    pub token: String,
    pub file_path: String,
    pub timeout: Duration,
    pub max_bytes: u64,
}

#[derive(Clone, PartialEq)]
pub struct MultipartUpload {
    pub token: String,
    pub endpoint: String,
    pub data_payload: Value,
    pub file_field: String,
    pub file_name: String,
    pub file_bytes: Vec<u8>,
    pub content_type: String,
    pub timeout: Duration,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TelegramFileOutcome {
    Downloaded(Vec<u8>),
    HttpError { status_code: u16 },
    TransportError { kind: TransportFailureKind },
}

pub trait TelegramTransport {
    fn send(&self, request: &TelegramRequest) -> Result<HttpResponse, TransportFailureKind>;

    fn send_action_multipart(
        &self,
        _request: &TelegramMultipartRequest,
    ) -> Result<HttpResponse, TransportFailureKind> {
        Err(TransportFailureKind::Request)
    }
}

pub trait TelegramMultipartTransport {
    fn send_multipart(
        &self,
        request: &TelegramMultipartRequest,
    ) -> Result<HttpResponse, TransportFailureKind>;
}

pub trait TelegramFileTransport {
    fn download(
        &self,
        request: &TelegramFileRequest,
    ) -> Result<BinaryHttpResponse, TransportFailureKind>;
}

pub struct ReqwestTelegramTransport {
    client: Client,
    api_base: String,
}

impl ReqwestTelegramTransport {
    pub fn new() -> Result<Self, TransportFailureKind> {
        Client::builder()
            .build()
            .map(|client| Self {
                client,
                api_base: TELEGRAM_API_BASE.to_owned(),
            })
            .map_err(|_| TransportFailureKind::Request)
    }

    #[cfg(test)]
    fn with_api_base(api_base: &str) -> Result<Self, TransportFailureKind> {
        Self::new().map(|mut transport| {
            transport.api_base = api_base.trim_end_matches('/').to_owned();
            transport
        })
    }
}

impl TelegramTransport for ReqwestTelegramTransport {
    fn send(&self, request: &TelegramRequest) -> Result<HttpResponse, TransportFailureKind> {
        let url = format!(
            "{}/bot{}/{}",
            self.api_base, request.token, request.endpoint
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
        read_response(builder.send().map_err(classify_error)?)
    }

    fn send_action_multipart(
        &self,
        request: &TelegramMultipartRequest,
    ) -> Result<HttpResponse, TransportFailureKind> {
        TelegramMultipartTransport::send_multipart(self, request)
    }
}

impl TelegramMultipartTransport for ReqwestTelegramTransport {
    fn send_multipart(
        &self,
        request: &TelegramMultipartRequest,
    ) -> Result<HttpResponse, TransportFailureKind> {
        let url = format!(
            "{}/bot{}/{}",
            self.api_base, request.token, request.endpoint
        );
        let mut form = Form::new();
        for (name, value) in &request.fields {
            form = form.text(name.clone(), value.clone());
        }
        let length =
            u64::try_from(request.file_bytes.len()).map_err(|_| TransportFailureKind::Request)?;
        let part = Part::reader_with_length(Cursor::new(request.file_bytes.clone()), length)
            .file_name(request.file_name.clone())
            .mime_str(&request.content_type)
            .map_err(|_| TransportFailureKind::Request)?;
        form = form.part(request.file_field.clone(), part);
        read_response(
            self.client
                .post(url)
                .timeout(request.timeout)
                .multipart(form)
                .send()
                .map_err(classify_error)?,
        )
    }
}

impl TelegramFileTransport for ReqwestTelegramTransport {
    fn download(
        &self,
        request: &TelegramFileRequest,
    ) -> Result<BinaryHttpResponse, TransportFailureKind> {
        let url = format!(
            "{}/file/bot{}/{}",
            self.api_base, request.token, request.file_path
        );
        let response = self
            .client
            .get(url)
            .timeout(request.timeout)
            .send()
            .map_err(classify_error)?;
        let status_code = response.status().as_u16();
        let body = read_limited(response, request.max_bytes)?;
        Ok(BinaryHttpResponse { status_code, body })
    }
}

fn read_limited(reader: impl Read, max_bytes: u64) -> Result<Vec<u8>, TransportFailureKind> {
    let mut body = Vec::new();
    reader
        .take(max_bytes.saturating_add(1))
        .read_to_end(&mut body)
        .map_err(|_| TransportFailureKind::Request)?;
    if body.len() as u64 > max_bytes {
        return Err(TransportFailureKind::ResponseTooLarge);
    }
    Ok(body)
}

fn read_response(
    response: reqwest::blocking::Response,
) -> Result<HttpResponse, TransportFailureKind> {
    let status_code = response.status().as_u16();
    let body = read_limited(response, MAX_RESPONSE_BYTES)?;
    Ok(HttpResponse {
        status_code,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
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

fn form_fields(payload: Value) -> Result<Vec<(String, String)>, TelegramHttpError> {
    let Value::Object(payload) = payload else {
        return Err(TelegramHttpError::InvalidPayload);
    };
    payload
        .into_iter()
        .filter_map(|(name, value)| match value {
            Value::Null => None,
            Value::String(value) => Some(Ok((name, value))),
            Value::Bool(true) => Some(Ok((name, "True".to_owned()))),
            Value::Bool(false) => Some(Ok((name, "False".to_owned()))),
            Value::Number(value) => Some(Ok((name, value.to_string()))),
            Value::Array(_) | Value::Object(_) => Some(Err(TelegramHttpError::InvalidPayload)),
        })
        .collect()
}

pub fn multipart_request_with<T: TelegramMultipartTransport>(
    transport: &T,
    upload: MultipartUpload,
) -> Result<TelegramHttpOutcome, TelegramHttpError> {
    if upload.timeout.is_zero() {
        return Err(TelegramHttpError::InvalidTimeout);
    }
    let request = TelegramMultipartRequest {
        token: upload.token,
        endpoint: upload.endpoint,
        fields: form_fields(upload.data_payload)?,
        file_field: upload.file_field,
        file_name: upload.file_name,
        file_bytes: upload.file_bytes.into(),
        content_type: upload.content_type,
        timeout: upload.timeout,
    };
    match transport.send_multipart(&request) {
        Ok(response) => Ok(response_outcome(response.status_code, response.body)),
        Err(kind) => Ok(TelegramHttpOutcome::TransportError { kind }),
    }
}

pub fn multipart_request(
    upload: MultipartUpload,
) -> Result<TelegramHttpOutcome, TelegramHttpError> {
    match ReqwestTelegramTransport::new() {
        Ok(transport) => multipart_request_with(&transport, upload),
        Err(kind) => Ok(TelegramHttpOutcome::TransportError { kind }),
    }
}

pub fn download_file_with<T: TelegramFileTransport>(
    transport: &T,
    token: &str,
    file_path: &str,
    timeout_seconds: u64,
) -> Result<TelegramFileOutcome, TelegramHttpError> {
    if timeout_seconds == 0 {
        return Err(TelegramHttpError::InvalidTimeout);
    }
    let request = TelegramFileRequest {
        token: token.to_owned(),
        file_path: file_path.to_owned(),
        timeout: Duration::from_secs(timeout_seconds),
        max_bytes: TELEGRAM_FILE_MAX_BYTES,
    };
    match transport.download(&request) {
        Ok(response) if (200..300).contains(&response.status_code) => {
            Ok(TelegramFileOutcome::Downloaded(response.body))
        }
        Ok(response) => Ok(TelegramFileOutcome::HttpError {
            status_code: response.status_code,
        }),
        Err(kind) => Ok(TelegramFileOutcome::TransportError { kind }),
    }
}

pub fn download_file(
    token: &str,
    file_path: &str,
    timeout_seconds: u64,
) -> Result<TelegramFileOutcome, TelegramHttpError> {
    match ReqwestTelegramTransport::new() {
        Ok(transport) => download_file_with(&transport, token, file_path, timeout_seconds),
        Err(kind) => Ok(TelegramFileOutcome::TransportError { kind }),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use reqwest::Method;
    use serde_json::json;

    use super::{
        BinaryHttpResponse, HttpResponse, MultipartUpload, ReqwestTelegramTransport,
        TELEGRAM_FILE_MAX_BYTES, TelegramFileOutcome, TelegramFileRequest, TelegramFileTransport,
        TelegramHttpError, TelegramHttpOutcome, TelegramMultipartRequest,
        TelegramMultipartTransport, TelegramRequest, TelegramTransport, TransportFailureKind,
        download_file_with, multipart_request_with, read_limited, request_with, response_outcome,
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

    struct FakeMultipartTransport {
        response: RefCell<Option<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<TelegramMultipartRequest>>,
    }

    impl TelegramMultipartTransport for FakeMultipartTransport {
        fn send_multipart(
            &self,
            request: &TelegramMultipartRequest,
        ) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            self.response
                .borrow_mut()
                .take()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    struct FakeFileTransport {
        response: RefCell<Option<Result<BinaryHttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<TelegramFileRequest>>,
    }

    impl TelegramFileTransport for FakeFileTransport {
        fn download(
            &self,
            request: &TelegramFileRequest,
        ) -> Result<BinaryHttpResponse, TransportFailureKind> {
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

    #[test]
    fn multipart_request_preserves_python_form_and_file_semantics() {
        let transport = FakeMultipartTransport {
            response: RefCell::new(Some(Ok(HttpResponse {
                status_code: 200,
                body: r#"{"ok":true}"#.to_owned(),
            }))),
            requests: RefCell::new(Vec::new()),
        };
        let actual = multipart_request_with(
            &transport,
            MultipartUpload {
                token: "synthetic-token".to_owned(),
                endpoint: "sendPhoto".to_owned(),
                data_payload: json!({"chat_id": "42", "reply_to_message_id": 7, "enabled": true, "skip": null}),
                file_field: "photo".to_owned(),
                file_name: "chart.png".to_owned(),
                file_bytes: vec![1, 2, 3],
                content_type: "image/png".to_owned(),
                timeout: Duration::from_secs(30),
            },
        );
        assert!(matches!(
            actual,
            Ok(TelegramHttpOutcome::Response {
                status_code: 200,
                ..
            })
        ));
        let requests = transport.requests.borrow();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].fields,
            vec![
                ("chat_id".to_owned(), "42".to_owned()),
                ("enabled".to_owned(), "True".to_owned()),
                ("reply_to_message_id".to_owned(), "7".to_owned()),
            ]
        );
        assert_eq!(requests[0].file_field, "photo");
        assert_eq!(requests[0].file_name, "chart.png");
        assert_eq!(requests[0].file_bytes.as_ref(), [1, 2, 3]);
        assert_eq!(requests[0].content_type, "image/png");
        assert_eq!(requests[0].timeout.as_secs(), 30);
    }

    #[test]
    fn multipart_rejects_nested_form_values_before_transport() {
        let transport = FakeMultipartTransport {
            response: RefCell::new(Some(Err(TransportFailureKind::Request))),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            multipart_request_with(
                &transport,
                MultipartUpload {
                    token: "token".to_owned(),
                    endpoint: "sendPhoto".to_owned(),
                    data_payload: json!({"nested": {"unsupported": true}}),
                    file_field: "photo".to_owned(),
                    file_name: "chart.png".to_owned(),
                    file_bytes: Vec::new(),
                    content_type: "image/png".to_owned(),
                    timeout: Duration::from_secs(5),
                },
            ),
            Err(TelegramHttpError::InvalidPayload)
        );
        assert!(transport.requests.borrow().is_empty());
    }

    #[test]
    fn multipart_handles_false_flags_and_all_validation_and_transport_boundaries() {
        let transport = FakeMultipartTransport {
            response: RefCell::new(Some(Err(TransportFailureKind::Connection))),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            multipart_request_with(
                &transport,
                MultipartUpload {
                    token: "synthetic-token".to_owned(),
                    endpoint: "sendDocument".to_owned(),
                    data_payload: json!({"enabled": false}),
                    file_field: "document".to_owned(),
                    file_name: "synthetic.bin".to_owned(),
                    file_bytes: vec![1],
                    content_type: "application/octet-stream".to_owned(),
                    timeout: Duration::from_secs(1),
                },
            ),
            Ok(TelegramHttpOutcome::TransportError {
                kind: TransportFailureKind::Connection,
            })
        );
        assert_eq!(
            transport.requests.borrow()[0].fields,
            [("enabled".to_owned(), "False".to_owned())]
        );

        for payload in [json!([]), json!({"nested": []})] {
            let transport = FakeMultipartTransport {
                response: RefCell::new(Some(Err(TransportFailureKind::Request))),
                requests: RefCell::new(Vec::new()),
            };
            let result = multipart_request_with(
                &transport,
                MultipartUpload {
                    token: "synthetic-token".to_owned(),
                    endpoint: "sendDocument".to_owned(),
                    data_payload: payload,
                    file_field: "document".to_owned(),
                    file_name: "synthetic.bin".to_owned(),
                    file_bytes: Vec::new(),
                    content_type: "application/octet-stream".to_owned(),
                    timeout: Duration::from_secs(1),
                },
            );
            assert_eq!(result, Err(TelegramHttpError::InvalidPayload));
            assert!(transport.requests.borrow().is_empty());
        }

        let transport = FakeMultipartTransport {
            response: RefCell::new(Some(Err(TransportFailureKind::Request))),
            requests: RefCell::new(Vec::new()),
        };
        let result = multipart_request_with(
            &transport,
            MultipartUpload {
                token: String::new(),
                endpoint: String::new(),
                data_payload: json!({}),
                file_field: String::new(),
                file_name: String::new(),
                file_bytes: Vec::new(),
                content_type: String::new(),
                timeout: Duration::ZERO,
            },
        );
        assert_eq!(result, Err(TelegramHttpError::InvalidTimeout));
        assert!(transport.requests.borrow().is_empty());
    }

    #[test]
    fn file_download_preserves_binary_content_and_request_identity() {
        let transport = FakeFileTransport {
            response: RefCell::new(Some(Ok(BinaryHttpResponse {
                status_code: 200,
                body: vec![0, 127, 255],
            }))),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            download_file_with(&transport, "synthetic-token", "photos/file_123.jpg", 30,),
            Ok(TelegramFileOutcome::Downloaded(vec![0, 127, 255]))
        );
        let requests = transport.requests.borrow();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].token, "synthetic-token");
        assert_eq!(requests[0].file_path, "photos/file_123.jpg");
        assert_eq!(requests[0].timeout.as_secs(), 30);
        assert_eq!(requests[0].max_bytes, TELEGRAM_FILE_MAX_BYTES);
    }

    #[test]
    fn bounded_reader_rejects_oversized_responses() {
        assert_eq!(
            read_limited(std::io::Cursor::new(vec![1, 2, 3]), 3),
            Ok(vec![1, 2, 3])
        );
        assert_eq!(
            read_limited(std::io::Cursor::new(vec![1, 2, 3, 4]), 3),
            Err(TransportFailureKind::ResponseTooLarge)
        );
    }

    #[test]
    fn file_download_classifies_http_transport_and_validation_failures() {
        let http_error = FakeFileTransport {
            response: RefCell::new(Some(Ok(BinaryHttpResponse {
                status_code: 404,
                body: b"not found".to_vec(),
            }))),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            download_file_with(&http_error, "token", "missing", 30),
            Ok(TelegramFileOutcome::HttpError { status_code: 404 })
        );

        let transport_error = FakeFileTransport {
            response: RefCell::new(Some(Err(TransportFailureKind::Timeout))),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            download_file_with(&transport_error, "token", "slow", 30),
            Ok(TelegramFileOutcome::TransportError {
                kind: TransportFailureKind::Timeout,
            })
        );

        let invalid = FakeFileTransport {
            response: RefCell::new(Some(Err(TransportFailureKind::Request))),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            download_file_with(&invalid, "token", "file", 0),
            Err(TelegramHttpError::InvalidTimeout)
        );
        assert!(invalid.requests.borrow().is_empty());
    }

    #[test]
    fn reqwest_transport_covers_json_multipart_and_binary_http_boundaries() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|_| unreachable!());
        let address = listener.local_addr().unwrap_or_else(|_| unreachable!());
        let server = thread::spawn(move || {
            for (expected_line, body) in [
                (
                    "POST /bot-token/sendMessage?query=value HTTP/1.1",
                    b"json".as_slice(),
                ),
                (
                    "POST /bot-token/sendPhoto HTTP/1.1",
                    b"multipart".as_slice(),
                ),
                (
                    "GET /file/bot-token/photos/file.bin HTTP/1.1",
                    &[0_u8, 127, 255],
                ),
            ] {
                let (mut stream, _) = listener.accept().unwrap_or_else(|_| unreachable!());
                let mut request = [0_u8; 16_384];
                let bytes = stream.read(&mut request).unwrap_or_default();
                let request = String::from_utf8_lossy(&request[..bytes]);
                assert!(request.starts_with(expected_line), "{request}");
                if expected_line.contains("sendMessage") {
                    assert!(request.contains(r#"{"text":"synthetic"}"#));
                }
                if expected_line.contains("sendPhoto") {
                    assert!(request.contains("name=\"chat_id\""));
                    assert!(request.contains("synthetic-photo"));
                }
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap_or_else(|_| unreachable!());
                stream.write_all(body).unwrap_or_else(|_| unreachable!());
            }
        });
        let reqwest_transport =
            ReqwestTelegramTransport::with_api_base(&format!("http://{address}"))
                .unwrap_or_else(|_| unreachable!());
        let response = reqwest_transport
            .send(&TelegramRequest {
                token: "-token".to_owned(),
                endpoint: "sendMessage".to_owned(),
                method: Method::POST,
                params: Some(json!({"query":"value"})),
                json_payload: Some(json!({"text":"synthetic"})),
                timeout: Duration::from_secs(5),
            })
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(response.body, "json");
        let response = TelegramMultipartTransport::send_multipart(
            &reqwest_transport,
            &TelegramMultipartRequest {
                token: "-token".to_owned(),
                endpoint: "sendPhoto".to_owned(),
                fields: vec![("chat_id".to_owned(), "42".to_owned())],
                file_field: "photo".to_owned(),
                file_name: "synthetic.png".to_owned(),
                file_bytes: Arc::from(b"synthetic-photo".as_slice()),
                content_type: "image/png".to_owned(),
                timeout: Duration::from_secs(5),
            },
        )
        .unwrap_or_else(|_| unreachable!());
        assert_eq!(response.body, "multipart");
        let response = reqwest_transport
            .download(&TelegramFileRequest {
                token: "-token".to_owned(),
                file_path: "photos/file.bin".to_owned(),
                timeout: Duration::from_secs(5),
                max_bytes: 3,
            })
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(response.body, [0, 127, 255]);
        assert!(server.join().is_ok());

        let fake = transport(Ok(HttpResponse {
            status_code: 200,
            body: String::new(),
        }));
        assert_eq!(
            fake.send_action_multipart(&TelegramMultipartRequest {
                token: String::new(),
                endpoint: String::new(),
                fields: Vec::new(),
                file_field: String::new(),
                file_name: String::new(),
                file_bytes: Arc::from([]),
                content_type: String::new(),
                timeout: Duration::from_secs(1),
            }),
            Err(TransportFailureKind::Request)
        );
    }
}
