//! Telegram `sendPoll` for polls the AI creates.
//!
//! This bypasses the action sink because the bot needs the returned poll id
//! to match later `poll_answer` updates, and action receipts only carry the
//! message id.

use std::time::Duration;

use reqwest::Method;
use serde_json::{Value, json};
use thiserror::Error;

use bot_core::polls::PollRequest;

use crate::telegram_http::{
    TelegramHttpOutcome, TelegramRequest, TelegramTransport, TransportFailureKind, send_with,
};

const SEND_POLL_TIMEOUT_SECONDS: u64 = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SentPoll {
    pub message_id: i64,
    pub poll_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SendPollError {
    /// The poll may or may not have been posted.
    #[error("Telegram sendPoll transport failed: {0:?}")]
    Transport(TransportFailureKind),
    #[error("Telegram rejected sendPoll ({status_code}): {description}")]
    Rejected {
        status_code: u16,
        description: String,
    },
    #[error("Telegram sendPoll response was malformed")]
    InvalidResponse,
}

#[must_use]
pub fn send_poll_payload(
    chat_id: i64,
    reply_to_message_id: Option<i64>,
    poll: &PollRequest,
) -> Value {
    let mut payload = json!({
        "chat_id": chat_id,
        "question": poll.question,
        "options": poll.options.iter().map(|text| json!({"text": text})).collect::<Vec<_>>(),
        "is_anonymous": poll.anonymous,
        "allows_multiple_answers": poll.multiple_answers,
    });
    if let Some(message_id) = reply_to_message_id {
        payload["reply_parameters"] =
            json!({"message_id": message_id, "allow_sending_without_reply": true});
    }
    payload
}

pub fn send_poll_with<T: TelegramTransport>(
    transport: &T,
    token: &str,
    chat_id: i64,
    reply_to_message_id: Option<i64>,
    poll: &PollRequest,
) -> Result<SentPoll, SendPollError> {
    let request = TelegramRequest {
        token: token.to_owned(),
        endpoint: "sendPoll".to_owned(),
        method: Method::POST,
        params: None,
        json_payload: Some(send_poll_payload(chat_id, reply_to_message_id, poll)),
        timeout: Duration::from_secs(SEND_POLL_TIMEOUT_SECONDS),
    };
    let (status_code, body) = match send_with(transport, &request) {
        TelegramHttpOutcome::Response { status_code, body } => (status_code, body),
        TelegramHttpOutcome::TransportError { kind } => {
            return Err(SendPollError::Transport(kind));
        }
    };
    let envelope = serde_json::from_str::<Value>(&body).unwrap_or(Value::Null);
    if !(200..300).contains(&status_code) || envelope.get("ok") != Some(&Value::Bool(true)) {
        return Err(SendPollError::Rejected {
            status_code,
            description: envelope
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("telegram request failed")
                .to_owned(),
        });
    }
    let result = envelope.get("result");
    let message_id = result
        .and_then(|result| result.get("message_id"))
        .and_then(Value::as_i64);
    let poll_id = result
        .and_then(|result| result.get("poll"))
        .and_then(|poll| poll.get("id"))
        .and_then(Value::as_str);
    match (message_id, poll_id) {
        (Some(message_id), Some(poll_id)) => Ok(SentPoll {
            message_id,
            poll_id: poll_id.to_owned(),
        }),
        _ => Err(SendPollError::InvalidResponse),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::telegram_http::HttpResponse;

    struct Transport {
        reply: Result<HttpResponse, TransportFailureKind>,
        requests: RefCell<Vec<TelegramRequest>>,
    }

    impl TelegramTransport for Transport {
        fn send(&self, request: &TelegramRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            self.reply.clone()
        }
    }

    fn transport(status_code: u16, body: &str) -> Transport {
        Transport {
            reply: Ok(HttpResponse {
                status_code,
                body: body.to_owned(),
            }),
            requests: RefCell::new(Vec::new()),
        }
    }

    fn poll() -> PollRequest {
        PollRequest {
            question: "¿Asado?".to_owned(),
            options: vec!["Sí".to_owned(), "No".to_owned()],
            anonymous: false,
            multiple_answers: true,
        }
    }

    #[test]
    fn sends_a_public_poll_as_a_reply_and_returns_its_ids() {
        let poll = poll();
        let transport = transport(
            200,
            r#"{"ok":true,"result":{"message_id":9,"poll":{"id":"poll-1"}}}"#,
        );
        assert_eq!(
            send_poll_with(&transport, "synthetic-token", -100, Some(4), &poll),
            Ok(SentPoll {
                message_id: 9,
                poll_id: "poll-1".to_owned(),
            })
        );
        let requests = transport.requests.borrow();
        assert_eq!(requests[0].endpoint, "sendPoll");
        assert_eq!(requests[0].method, Method::POST);
        assert_eq!(
            requests[0].json_payload,
            Some(json!({
                "chat_id": -100,
                "question": "¿Asado?",
                "options": [{"text": "Sí"}, {"text": "No"}],
                "is_anonymous": false,
                "allows_multiple_answers": true,
                "reply_parameters": {"message_id": 4, "allow_sending_without_reply": true}
            }))
        );
        assert!(
            send_poll_payload(1, None, &poll)
                .get("reply_parameters")
                .is_none()
        );
    }

    #[test]
    fn reports_rejections_malformed_bodies_and_transport_failures() {
        let poll = poll();
        assert_eq!(
            send_poll_with(
                &transport(
                    400,
                    r#"{"ok":false,"description":"Bad Request: polls can't be sent"}"#
                ),
                "t",
                1,
                None,
                &poll
            ),
            Err(SendPollError::Rejected {
                status_code: 400,
                description: "Bad Request: polls can't be sent".to_owned(),
            })
        );
        assert_eq!(
            send_poll_with(&transport(502, "<html>"), "t", 1, None, &poll),
            Err(SendPollError::Rejected {
                status_code: 502,
                description: "telegram request failed".to_owned(),
            })
        );
        assert_eq!(
            send_poll_with(
                &transport(200, r#"{"ok":true,"result":{"message_id":9}}"#),
                "t",
                1,
                None,
                &poll
            ),
            Err(SendPollError::InvalidResponse)
        );
        let timeout = Transport {
            reply: Err(TransportFailureKind::Timeout),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            send_poll_with(&timeout, "t", 1, None, &poll),
            Err(SendPollError::Transport(TransportFailureKind::Timeout))
        );
    }
}
