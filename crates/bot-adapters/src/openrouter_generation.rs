//! Blocking OpenRouter generation lookup used by billing reconciliation.

use std::time::Duration;

use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::{Map, Value};
use thiserror::Error;

const GENERATION_URL: &str = "https://openrouter.ai/api/v1/generation";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationRequest {
    pub generation_id: String,
    pub api_key: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status_code: u16,
    pub body: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GenerationError {
    #[error("OpenRouter generation transport failed: {0}")]
    Transport(String),
    #[error("OpenRouter generation HTTP {status_code}")]
    Http { status_code: u16 },
    #[error("OpenRouter generation returned invalid JSON: {0}")]
    InvalidJson(String),
}

pub trait GenerationTransport {
    fn get(&self, request: &GenerationRequest) -> Result<HttpResponse, GenerationError>;
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum GenerationOutcome {
    Pending,
    Success { generation: Map<String, Value> },
}

pub struct ReqwestGenerationTransport {
    client: Client,
}

impl ReqwestGenerationTransport {
    pub fn new() -> Result<Self, GenerationError> {
        Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(20))
            .build()
            .map(|client| Self { client })
            .map_err(|error| GenerationError::Transport(error.to_string()))
    }
}

impl GenerationTransport for ReqwestGenerationTransport {
    fn get(&self, request: &GenerationRequest) -> Result<HttpResponse, GenerationError> {
        let response = self
            .client
            .get(GENERATION_URL)
            .query(&[("id", &request.generation_id)])
            .bearer_auth(&request.api_key)
            .send()
            .map_err(|error| GenerationError::Transport(error.to_string()))?;
        let status_code = response.status().as_u16();
        response
            .text()
            .map(|body| HttpResponse { status_code, body })
            .map_err(|error| GenerationError::Transport(error.to_string()))
    }
}

pub fn parse_response(response: HttpResponse) -> Result<GenerationOutcome, GenerationError> {
    if response.status_code == 404 {
        return Ok(GenerationOutcome::Pending);
    }
    if response.status_code >= 400 {
        return Err(GenerationError::Http {
            status_code: response.status_code,
        });
    }
    let payload: Value = serde_json::from_str(&response.body)
        .map_err(|error| GenerationError::InvalidJson(error.to_string()))?;
    let Some(payload) = payload.as_object() else {
        return Ok(GenerationOutcome::Pending);
    };
    let generation = payload
        .get("data")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_else(|| payload.clone());
    Ok(GenerationOutcome::Success { generation })
}

pub fn fetch_with<T: GenerationTransport>(
    transport: &T,
    api_key: &str,
    generation_id: &str,
) -> Result<GenerationOutcome, GenerationError> {
    let request = GenerationRequest {
        generation_id: generation_id.to_owned(),
        api_key: api_key.to_owned(),
    };
    parse_response(transport.get(&request)?)
}

pub fn fetch(api_key: &str, generation_id: &str) -> Result<GenerationOutcome, GenerationError> {
    fetch_with(&ReqwestGenerationTransport::new()?, api_key, generation_id)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use serde_json::json;

    use super::{
        GenerationError, GenerationOutcome, GenerationRequest, GenerationTransport, HttpResponse,
        fetch_with, parse_response,
    };

    struct FakeTransport {
        response: RefCell<Option<Result<HttpResponse, GenerationError>>>,
        requests: RefCell<Vec<GenerationRequest>>,
    }

    impl GenerationTransport for FakeTransport {
        fn get(&self, request: &GenerationRequest) -> Result<HttpResponse, GenerationError> {
            self.requests.borrow_mut().push(request.clone());
            self.response
                .borrow_mut()
                .take()
                .unwrap_or_else(|| Err(GenerationError::Transport("missing response".to_owned())))
        }
    }

    fn response(status_code: u16, body: &str) -> HttpResponse {
        HttpResponse {
            status_code,
            body: body.to_owned(),
        }
    }

    #[test]
    fn nested_generation_is_returned_without_the_envelope() {
        let actual = parse_response(response(
            200,
            r#"{"data":{"id":"generation-1","total_cost":0.2},"meta":"ignored"}"#,
        ));
        assert_eq!(
            actual,
            Ok(GenerationOutcome::Success {
                generation: json!({"id": "generation-1", "total_cost": 0.2})
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
            })
        );
    }

    #[test]
    fn flat_generation_payload_remains_compatible() {
        let actual = parse_response(response(302, r#"{"id":"generation-2"}"#));
        assert_eq!(
            actual,
            Ok(GenerationOutcome::Success {
                generation: json!({"id": "generation-2"})
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
            })
        );
    }

    #[test]
    fn missing_and_non_object_generations_are_pending() {
        assert_eq!(
            parse_response(response(404, "not-json")),
            Ok(GenerationOutcome::Pending)
        );
        assert_eq!(
            parse_response(response(200, "[]")),
            Ok(GenerationOutcome::Pending)
        );
    }

    #[test]
    fn invalid_json_and_http_failures_are_typed() {
        assert!(matches!(
            parse_response(response(200, "not-json")),
            Err(GenerationError::InvalidJson(_))
        ));
        assert_eq!(
            parse_response(response(500, r#"{"error":"unavailable"}"#)),
            Err(GenerationError::Http { status_code: 500 })
        );
    }

    #[test]
    fn fetch_passes_credentials_and_identity_to_the_transport() {
        let transport = FakeTransport {
            response: RefCell::new(Some(Ok(response(404, "")))),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            fetch_with(&transport, "synthetic-key", "generation-3"),
            Ok(GenerationOutcome::Pending)
        );
        assert_eq!(
            transport.requests.borrow().as_slice(),
            &[GenerationRequest {
                generation_id: "generation-3".to_owned(),
                api_key: "synthetic-key".to_owned(),
            }]
        );
    }

    #[test]
    fn transport_failures_are_not_reclassified_as_pending() {
        let transport = FakeTransport {
            response: RefCell::new(Some(Err(GenerationError::Transport(
                "synthetic failure".to_owned(),
            )))),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            fetch_with(&transport, "key", "generation"),
            Err(GenerationError::Transport("synthetic failure".to_owned()))
        );
    }
}
