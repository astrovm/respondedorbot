//! Blocking OpenNode adapter for Lightning top-up charges.

use std::sync::OnceLock;
use std::time::Duration;

use reqwest::blocking::Client;
use serde_json::{Value, json};
use thiserror::Error;

pub use crate::http_client::TransportFailureKind;
use crate::http_client::classify_error;

pub const OPENNODE_API_URL: &str = "https://api.opennode.com";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenNodeRequest {
    /// `POST /v1/charges` with a JSON body.
    CreateCharge { body: Value },
    /// `GET /v2/charge/{id}`.
    ChargeInfo { charge_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status_code: u16,
    pub body: String,
}

pub trait OpenNodeTransport {
    fn send(&self, request: &OpenNodeRequest) -> Result<HttpResponse, TransportFailureKind>;
}

pub struct ReqwestOpenNodeTransport {
    client: Client,
    base_url: String,
    api_key: String,
}

impl ReqwestOpenNodeTransport {
    pub fn new(base_url: &str, api_key: &str) -> Result<Self, TransportFailureKind> {
        static CLIENT: OnceLock<Client> = OnceLock::new();
        crate::http_client::shared_client(&CLIENT, || {
            Client::builder().timeout(REQUEST_TIMEOUT).build()
        })
        .map(|client| Self {
            client,
            base_url: base_url.trim_end_matches('/').to_owned(),
            api_key: api_key.to_owned(),
        })
        .map_err(|_| TransportFailureKind::Request)
    }
}

impl OpenNodeTransport for ReqwestOpenNodeTransport {
    fn send(&self, request: &OpenNodeRequest) -> Result<HttpResponse, TransportFailureKind> {
        let builder = match request {
            OpenNodeRequest::CreateCharge { body } => self
                .client
                .post(format!("{}/v1/charges", self.base_url))
                .json(body),
            OpenNodeRequest::ChargeInfo { charge_id } => self
                .client
                .get(format!("{}/v2/charge/{charge_id}", self.base_url)),
        };
        let response = builder
            .header("Authorization", &self.api_key)
            .send()
            .map_err(classify_error)?;
        let status_code = response.status().as_u16();
        response
            .text()
            .map(|body| HttpResponse { status_code, body })
            .map_err(classify_error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OpenNodeError {
    #[error("OpenNode request failed: {0:?}")]
    Transport(TransportFailureKind),
    #[error("OpenNode returned HTTP {status_code}: {message}")]
    Http { status_code: u16, message: String },
    #[error("OpenNode returned invalid JSON")]
    InvalidJson,
    #[error("OpenNode response is missing {0}")]
    InvalidPayload(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewCharge {
    pub usd_cents: i64,
    pub description: String,
    pub order_id: String,
    pub ttl_minutes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LightningCharge {
    pub id: String,
    pub payreq: String,
    pub checkout_url: Option<String>,
    pub sats: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChargeStatus {
    Pending,
    Paid,
    /// Expired, or refunded after an on-chain underpayment: never credited.
    Closed,
}

fn data(response: HttpResponse) -> Result<Value, OpenNodeError> {
    let payload = serde_json::from_str::<Value>(&response.body);
    if response.status_code >= 400 {
        let message = payload
            .ok()
            .and_then(|payload| {
                payload
                    .get("message")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .unwrap_or_default();
        return Err(OpenNodeError::Http {
            status_code: response.status_code,
            message,
        });
    }
    payload
        .map_err(|_| OpenNodeError::InvalidJson)?
        .get_mut("data")
        .map(Value::take)
        .ok_or(OpenNodeError::InvalidPayload("data"))
}

fn text<'a>(value: &'a Value, field: &'static str) -> Result<&'a str, OpenNodeError> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .ok_or(OpenNodeError::InvalidPayload(field))
}

/// USD amounts go out as a decimal number of dollars, built from integer
/// cents so no rounding happens on our side.
fn usd_amount(cents: i64) -> Value {
    let text = format!("{}.{:02}", cents / 100, cents % 100);
    serde_json::from_str(&text).unwrap_or(Value::Null)
}

pub fn create_charge<T: OpenNodeTransport>(
    transport: &T,
    charge: &NewCharge,
) -> Result<LightningCharge, OpenNodeError> {
    let body = json!({
        "amount": usd_amount(charge.usd_cents),
        "currency": "USD",
        "description": charge.description,
        "order_id": charge.order_id,
        "ttl": charge.ttl_minutes,
    });
    let response = transport
        .send(&OpenNodeRequest::CreateCharge { body })
        .map_err(OpenNodeError::Transport)?;
    let data = data(response)?;
    let invoice = data
        .get("lightning_invoice")
        .ok_or(OpenNodeError::InvalidPayload("lightning_invoice"))?;
    Ok(LightningCharge {
        id: text(&data, "id")?.to_owned(),
        payreq: text(invoice, "payreq")?.to_owned(),
        checkout_url: text(&data, "hosted_checkout_url")
            .ok()
            .map(ToOwned::to_owned),
        sats: data.get("amount").and_then(Value::as_i64),
    })
}

pub fn charge_status<T: OpenNodeTransport>(
    transport: &T,
    charge_id: &str,
) -> Result<ChargeStatus, OpenNodeError> {
    let response = transport
        .send(&OpenNodeRequest::ChargeInfo {
            charge_id: charge_id.to_owned(),
        })
        .map_err(OpenNodeError::Transport)?;
    let data = data(response)?;
    Ok(match text(&data, "status")? {
        "paid" => ChargeStatus::Paid,
        "expired" | "refunded" => ChargeStatus::Closed,
        _ => ChargeStatus::Pending,
    })
}

#[cfg(test)]
mod tests {
    type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

    use std::cell::RefCell;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use serde_json::json;

    use super::{
        ChargeStatus, HttpResponse, LightningCharge, NewCharge, OpenNodeError, OpenNodeRequest,
        OpenNodeTransport, ReqwestOpenNodeTransport, TransportFailureKind, charge_status,
        create_charge, usd_amount,
    };

    struct FakeTransport {
        response: RefCell<Option<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<OpenNodeRequest>>,
    }

    impl FakeTransport {
        fn answering(status_code: u16, body: &str) -> Self {
            Self {
                response: RefCell::new(Some(Ok(HttpResponse {
                    status_code,
                    body: body.to_owned(),
                }))),
                requests: RefCell::new(Vec::new()),
            }
        }
    }

    impl OpenNodeTransport for FakeTransport {
        fn send(&self, request: &OpenNodeRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            self.response
                .borrow_mut()
                .take()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    fn charge() -> NewCharge {
        NewCharge {
            usd_cents: 133,
            description: "50 AI credits".to_owned(),
            order_id: "ln:p50:88".to_owned(),
            ttl_minutes: 30,
        }
    }

    #[test]
    fn usd_amounts_are_exact_decimal_dollars() {
        assert_eq!(usd_amount(33), json!(0.33));
        assert_eq!(usd_amount(1_625), json!(16.25));
        assert_eq!(usd_amount(500), json!(5.0));
    }

    #[test]
    fn creates_usd_charges_and_reads_the_bolt11_invoice() {
        let transport = FakeTransport::answering(
            201,
            r#"{"data":{"id":"charge-1","amount":512,"hosted_checkout_url":"https://checkout.example.test/charge-1","lightning_invoice":{"payreq":"lnbc512synthetic","expires_at":1}}}"#,
        );
        assert_eq!(
            create_charge(&transport, &charge()),
            Ok(LightningCharge {
                id: "charge-1".to_owned(),
                payreq: "lnbc512synthetic".to_owned(),
                checkout_url: Some("https://checkout.example.test/charge-1".to_owned()),
                sats: Some(512),
            })
        );
        assert_eq!(
            transport.requests.borrow().as_slice(),
            [OpenNodeRequest::CreateCharge {
                body: json!({
                    "amount": 1.33,
                    "currency": "USD",
                    "description": "50 AI credits",
                    "order_id": "ln:p50:88",
                    "ttl": 30,
                }),
            }]
        );
        let bare = FakeTransport::answering(
            200,
            r#"{"data":{"id":"charge-2","hosted_checkout_url":"","lightning_invoice":{"payreq":"lnbc"}}}"#,
        );
        assert_eq!(
            create_charge(&bare, &charge()).map(|charge| (charge.checkout_url, charge.sats)),
            Ok((None, None))
        );
    }

    #[test]
    fn charge_creation_failures_are_typed() {
        for (status_code, body, expected) in [
            (
                400,
                r#"{"success":false,"message":"Invalid amount"}"#,
                OpenNodeError::Http {
                    status_code: 400,
                    message: "Invalid amount".to_owned(),
                },
            ),
            (
                502,
                "bad gateway",
                OpenNodeError::Http {
                    status_code: 502,
                    message: String::new(),
                },
            ),
            (200, "not-json", OpenNodeError::InvalidJson),
            (200, "{}", OpenNodeError::InvalidPayload("data")),
            (
                200,
                r#"{"data":{"id":"x"}}"#,
                OpenNodeError::InvalidPayload("lightning_invoice"),
            ),
            (
                200,
                r#"{"data":{"lightning_invoice":{"payreq":"lnbc"}}}"#,
                OpenNodeError::InvalidPayload("id"),
            ),
            (
                200,
                r#"{"data":{"id":"x","lightning_invoice":{"payreq":""}}}"#,
                OpenNodeError::InvalidPayload("payreq"),
            ),
        ] {
            let transport = FakeTransport::answering(status_code, body);
            assert_eq!(
                create_charge(&transport, &charge()),
                Err(expected),
                "{body}"
            );
        }
        let offline = FakeTransport {
            response: RefCell::new(Some(Err(TransportFailureKind::Timeout))),
            requests: RefCell::new(Vec::new()),
        };
        let error = create_charge(&offline, &charge());
        assert_eq!(
            error,
            Err(OpenNodeError::Transport(TransportFailureKind::Timeout))
        );
        assert_eq!(
            error.map_err(|error| error.to_string()),
            Err("OpenNode request failed: Timeout".to_owned())
        );
    }

    #[test]
    fn statuses_map_to_pending_paid_or_closed() {
        for (status, expected) in [
            ("paid", ChargeStatus::Paid),
            ("expired", ChargeStatus::Closed),
            ("refunded", ChargeStatus::Closed),
            ("unpaid", ChargeStatus::Pending),
            ("processing", ChargeStatus::Pending),
            ("underpaid", ChargeStatus::Pending),
        ] {
            let body = format!(r#"{{"data":{{"status":"{status}"}}}}"#);
            let transport = FakeTransport::answering(200, &body);
            assert_eq!(charge_status(&transport, "charge-1"), Ok(expected));
            assert_eq!(
                transport.requests.borrow().as_slice(),
                [OpenNodeRequest::ChargeInfo {
                    charge_id: "charge-1".to_owned()
                }]
            );
        }
        let missing = FakeTransport::answering(200, r#"{"data":{}}"#);
        assert_eq!(
            charge_status(&missing, "charge-1"),
            Err(OpenNodeError::InvalidPayload("status"))
        );
        let offline = FakeTransport {
            response: RefCell::new(None),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            charge_status(&offline, "charge-1"),
            Err(OpenNodeError::Transport(TransportFailureKind::Request))
        );
    }

    /// Serves one canned response and returns the raw request it received.
    fn serve_once(body: &'static str) -> TestResult<(String, thread::JoinHandle<String>)> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || {
            listener
                .incoming()
                .flatten()
                .next()
                .map(|mut stream| {
                    let mut request = [0_u8; 4_096];
                    let bytes = stream.read(&mut request).unwrap_or_default();
                    let _written = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    String::from_utf8_lossy(&request[..bytes]).into_owned()
                })
                .unwrap_or_default()
        });
        Ok((format!("http://{address}/"), server))
    }

    #[test]
    fn reqwest_transport_authenticates_both_endpoints() -> TestResult {
        let (url, server) = serve_once(r#"{"data":{}}"#)?;
        let transport = ReqwestOpenNodeTransport::new(&url, "synthetic-key")
            .ok()
            .ok_or("unexpected error")?;
        let response = transport
            .send(&OpenNodeRequest::CreateCharge {
                body: json!({"amount": 1}),
            })
            .ok()
            .ok_or("unexpected error")?;
        assert_eq!(response.status_code, 200);
        assert_eq!(response.body, r#"{"data":{}}"#);
        let request = server.join().map_err(|_| "server panicked")?;
        assert!(request.starts_with("POST /v1/charges "), "{request}");
        assert!(
            request
                .to_ascii_lowercase()
                .contains("authorization: synthetic-key")
        );
        assert!(request.ends_with(r#"{"amount":1}"#), "{request}");

        let (url, server) = serve_once(r#"{"data":{"status":"paid"}}"#)?;
        let transport = ReqwestOpenNodeTransport::new(&url, "synthetic-key")
            .ok()
            .ok_or("unexpected error")?;
        assert_eq!(
            charge_status(&transport, "charge-1"),
            Ok(ChargeStatus::Paid)
        );
        let request = server.join().map_err(|_| "server panicked")?;
        assert!(request.starts_with("GET /v2/charge/charge-1 "), "{request}");

        let unavailable = ReqwestOpenNodeTransport::new("http://127.0.0.1:1", "synthetic-key")
            .ok()
            .ok_or("unexpected error")?;
        assert!(matches!(
            charge_status(&unavailable, "charge-1"),
            Err(OpenNodeError::Transport(_))
        ));
        Ok(())
    }
}
