//! Telegram `getUpdates` request and response handling.
//!
//! This module owns the untrusted JSON boundary. Callers receive an update id
//! and one of the update kinds supported by the bot runtime.

use std::time::Duration;

use reqwest::Method;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use thiserror::Error;

use bot_core::telegram_input::{
    ChatId, MessageContent, MessageId, UserId, extract_message_content, normalize_numeric_id,
};

use crate::telegram_http::{
    TelegramHttpError, TelegramHttpOutcome, TelegramRequest, TelegramTransport,
    TransportFailureKind, send_with,
};

pub const DEFAULT_LONG_POLL_SECONDS: u64 = 30;
const HTTP_TIMEOUT_MARGIN_SECONDS: u64 = 15;
const POLL_BATCH_LIMIT: u8 = 100;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IncomingUpdate {
    pub update_id: i64,
    pub event: IncomingEvent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IncomingEvent {
    Message(Box<IncomingMessage>),
    SuccessfulPayment(Map<String, Value>),
    CallbackQuery(Map<String, Value>),
    PreCheckoutQuery(Map<String, Value>),
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IncomingMessage {
    pub message_id: Option<MessageId>,
    pub chat_id: Option<ChatId>,
    pub chat_type: Option<String>,
    pub chat_title: Option<String>,
    pub sender_id: Option<UserId>,
    pub sender_first_name: Option<String>,
    pub sender_last_name: Option<String>,
    pub sender_username: Option<String>,
    pub sender_language_code: Option<String>,
    pub has_reply: bool,
    pub replied_message_id: Option<MessageId>,
    pub replied_sender_first_name: Option<String>,
    pub replied_sender_username: Option<String>,
    pub replied_text: Option<String>,
    pub visual_media_kind: Option<String>,
    pub audio_media_kind: Option<String>,
    pub audio_duration_seconds: Option<u64>,
    pub content: Option<MessageContent>,
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
        "message" if payload.contains_key("successful_payment") => {
            IncomingEvent::SuccessfulPayment(payload)
        }
        "message" => IncomingEvent::Message(Box::new(parse_message(&payload))),
        "callback_query" => IncomingEvent::CallbackQuery(payload),
        _ => IncomingEvent::PreCheckoutQuery(payload),
    })
}

fn parse_message(payload: &Map<String, Value>) -> IncomingMessage {
    let message_id = payload
        .get("message_id")
        .and_then(normalize_numeric_id)
        .map(MessageId);
    let chat = payload.get("chat").and_then(Value::as_object);
    let chat_id = chat
        .and_then(|chat| chat.get("id"))
        .and_then(normalize_numeric_id)
        .map(ChatId);
    let chat_type = chat
        .and_then(|chat| chat.get("type"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let chat_title = chat
        .and_then(|chat| chat.get("title"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let sender = payload.get("from").and_then(Value::as_object);
    let sender_id = sender
        .and_then(|sender| sender.get("id"))
        .and_then(normalize_numeric_id)
        .map(UserId);
    let sender_language_code = sender
        .and_then(|sender| sender.get("language_code"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let sender_first_name = sender
        .and_then(|sender| sender.get("first_name"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let sender_username = sender
        .and_then(|sender| sender.get("username"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let sender_last_name = sender
        .and_then(|sender| sender.get("last_name"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let replied = payload.get("reply_to_message").and_then(Value::as_object);
    let replied_message_id = replied
        .and_then(|reply| reply.get("message_id"))
        .and_then(normalize_numeric_id)
        .map(MessageId);
    let replied_sender_username = replied
        .and_then(|reply| reply.get("from"))
        .and_then(Value::as_object)
        .and_then(|sender| sender.get("username"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let replied_sender_first_name = replied
        .and_then(|reply| reply.get("from"))
        .and_then(Value::as_object)
        .and_then(|sender| sender.get("first_name"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let replied_text = replied
        .and_then(|reply| extract_message_content(&Value::Object(reply.clone())).ok())
        .map(|content| content.text)
        .filter(|text| !text.is_empty());
    let visual_media_kind = media_kind(payload, &["photo", "sticker", "animation"]);
    let audio_media_kind = media_kind(payload, &["voice", "audio", "video", "video_note"]);
    let audio_duration_seconds = media_object(payload, &["voice", "audio", "video", "video_note"])
        .and_then(|media| media.get("duration"))
        .and_then(|duration| match duration {
            Value::Number(value) => value
                .as_u64()
                .or_else(|| value.as_f64().map(|value| value.max(0.0) as u64)),
            Value::String(value) => value.parse::<f64>().ok().map(|value| value.max(0.0) as u64),
            _ => None,
        });
    IncomingMessage {
        message_id,
        chat_id,
        chat_type,
        chat_title,
        sender_id,
        sender_first_name,
        sender_last_name,
        sender_username,
        sender_language_code,
        has_reply: payload.contains_key("reply_to_message"),
        replied_message_id,
        replied_sender_first_name,
        replied_sender_username,
        replied_text,
        visual_media_kind,
        audio_media_kind,
        audio_duration_seconds,
        content: extract_message_content(&Value::Object(payload.clone())).ok(),
    }
}

fn media_object<'a>(
    payload: &'a Map<String, Value>,
    kinds: &[&str],
) -> Option<&'a Map<String, Value>> {
    kinds
        .iter()
        .find_map(|kind| payload.get(*kind).and_then(Value::as_object))
        .or_else(|| {
            let replied = payload.get("reply_to_message")?.as_object()?;
            kinds
                .iter()
                .find_map(|kind| replied.get(*kind).and_then(Value::as_object))
        })
}

fn media_kind(payload: &Map<String, Value>, kinds: &[&str]) -> Option<String> {
    kinds
        .iter()
        .find(|kind| payload.get(**kind).is_some_and(|value| !value.is_null()))
        .or_else(|| {
            let replied = payload.get("reply_to_message")?.as_object()?;
            kinds
                .iter()
                .find(|kind| replied.get(**kind).is_some_and(|value| !value.is_null()))
        })
        .map(|kind| (*kind).to_owned())
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
        "limit": POLL_BATCH_LIMIT,
        "allowed_updates": ["message", "callback_query", "pre_checkout_query"]
    });
    if let Some(offset) = offset {
        params["offset"] = json!(offset);
    }
    let mut batch_limit = POLL_BATCH_LIMIT;
    loop {
        params["limit"] = json!(batch_limit);
        let request = TelegramRequest {
            token: token.to_owned(),
            endpoint: "getUpdates".to_owned(),
            method: Method::POST,
            params: None,
            json_payload: Some(params.clone()),
            timeout: Duration::from_secs(request_timeout),
        };
        match send_with(transport, &request) {
            TelegramHttpOutcome::Response { status_code, body } => {
                return parse_response(status_code, &body);
            }
            TelegramHttpOutcome::TransportError {
                kind: TransportFailureKind::ResponseTooLarge,
            } if batch_limit > 1 => {
                // Keep the same offset: acknowledge nothing until a smaller batch is parsed.
                // This avoids replaying an oversized head forever without weakening the cap.
                batch_limit = (batch_limit / 2).max(1);
            }
            TelegramHttpOutcome::TransportError { kind } => {
                return Ok(PollOutcome::Retry(PollFailure::Transport { failure: kind }));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

    use std::cell::RefCell;

    use bot_core::telegram_input::{MessageContent, MessageId};
    use reqwest::Method;
    use serde_json::{Value, json};

    use super::{
        IncomingEvent, IncomingMessage, IncomingUpdate, PollFailure, PollOutcome, PollingError,
        next_offset, parse_response, poll_once_with,
    };
    use crate::telegram_http::{
        HttpResponse, TelegramRequest, TelegramTransport, TransportFailureKind,
    };

    type Responder = Box<dyn Fn(&TelegramRequest) -> Result<HttpResponse, TransportFailureKind>>;

    /// One transport type for every polling scenario: either a single scripted
    /// result or a responder computing each reply from the request.
    struct FakeTransport {
        result: RefCell<Option<Result<HttpResponse, TransportFailureKind>>>,
        responder: Option<Responder>,
        requests: RefCell<Vec<TelegramRequest>>,
    }

    impl TelegramTransport for FakeTransport {
        fn send(&self, request: &TelegramRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            if let Some(responder) = &self.responder {
                return responder(request);
            }
            self.result
                .borrow_mut()
                .take()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    fn transport(result: Result<HttpResponse, TransportFailureKind>) -> FakeTransport {
        FakeTransport {
            result: RefCell::new(Some(result)),
            responder: None,
            requests: RefCell::new(Vec::new()),
        }
    }

    fn responding(responder: Responder) -> FakeTransport {
        FakeTransport {
            result: RefCell::new(None),
            responder: Some(responder),
            requests: RefCell::new(Vec::new()),
        }
    }

    fn batch_limit(request: &TelegramRequest) -> Option<u64> {
        request.json_payload.as_ref()?["limit"].as_u64()
    }

    #[test]
    fn poll_request_accepts_a_full_telegram_update_batch() {
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
        assert_eq!(requests[0].method, Method::POST);
        assert_eq!(requests[0].timeout.as_secs(), 45);
        assert_eq!(requests[0].params, None);
        assert_eq!(
            requests[0].json_payload,
            Some(json!({
                "offset": 42,
                "timeout": 30,
                "limit": 100,
                "allowed_updates": ["message", "callback_query", "pre_checkout_query"]
            }))
        );
    }

    #[test]
    fn oversized_multibyte_batches_shrink_without_acknowledging_updates()
    -> Result<(), Box<dyn std::error::Error>> {
        let transport = responding(Box::new(|request| {
            let limit = batch_limit(request).unwrap_or_default();
            let offset = request
                .json_payload
                .as_ref()
                .and_then(|params| params["offset"].as_i64())
                .unwrap_or_default();
            let updates = (0..limit)
                .map(|index| {
                    json!({
                        "update_id": offset + index as i64,
                        "message": {"message_id": index, "text": "🦀".repeat(4096),
                            "chat": {"id": 1, "type": "private"}}
                    })
                })
                .collect::<Vec<_>>();
            let body = json!({"ok": true, "result": updates}).to_string();
            (body.len() <= 1_048_576)
                .then_some(HttpResponse {
                    status_code: 200,
                    body,
                })
                .ok_or(TransportFailureKind::ResponseTooLarge)
        }));
        let updates = parsed_updates(poll_once_with(&transport, "token", Some(42), 30));
        assert_eq!(updates.len(), 50);
        assert_eq!(updates.first().ok_or("nonempty batch")?.update_id, 42);
        assert_eq!(next_offset(&updates, Some(42)), Some(92));
        let requests = transport.requests.borrow();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0]
                .json_payload
                .as_ref()
                .ok_or("polling JSON payload")?["limit"],
            100
        );
        assert_eq!(
            requests[1]
                .json_payload
                .as_ref()
                .ok_or("polling JSON payload")?["limit"],
            50
        );
        for request in requests.iter() {
            assert_eq!(
                request
                    .json_payload
                    .as_ref()
                    .ok_or("polling JSON payload")?["offset"],
                42
            );
        }
        Ok(())
    }

    #[test]
    fn an_oversized_single_update_stops_the_bounded_retry() {
        let transport = responding(Box::new(|_| Err(TransportFailureKind::ResponseTooLarge)));
        assert_eq!(
            poll_once_with(&transport, "token", Some(42), 30),
            Ok(PollOutcome::Retry(PollFailure::Transport {
                failure: TransportFailureKind::ResponseTooLarge
            }))
        );
        let requests = transport.requests.borrow();
        assert_eq!(
            requests.iter().map(batch_limit).collect::<Vec<_>>(),
            [100, 50, 25, 12, 6, 3, 1].map(Some)
        );
        assert!(requests.iter().all(|request| {
            request
                .json_payload
                .as_ref()
                .is_some_and(|params| params["offset"] == 42)
        }));
    }

    fn parsed_updates(result: Result<PollOutcome, PollingError>) -> Vec<IncomingUpdate> {
        match result {
            Ok(PollOutcome::Updates(updates)) => updates,
            Ok(PollOutcome::Retry(_)) | Err(_) => Vec::new(),
        }
    }

    fn message_event(event: &IncomingEvent) -> Option<&IncomingMessage> {
        match event {
            IncomingEvent::Message(message) => Some(message),
            _ => None,
        }
    }

    #[test]
    fn test_helpers_reject_retries_errors_and_non_message_events() {
        assert!(parsed_updates(Err(PollingError::InvalidResponse)).is_empty());
        assert!(parsed_updates(Ok(PollOutcome::Retry(PollFailure::Conflict))).is_empty());
        assert_eq!(message_event(&IncomingEvent::Unsupported), None);
    }

    #[test]
    fn response_decodes_supported_and_unsupported_updates() -> TestResult {
        let actual = parse_response(
            200,
            r#"{"ok":true,"result":[
                {"update_id":10,"message":{"message_id":1,"text":"hola","chat":{"id":-42,"type":"group","title":"Synthetic Group"},"from":{"id":8,"first_name":"Ana","last_name":"Test","username":"ana","language_code":"es"},"reply_to_message":{"message_id":3,"text":"earlier answer","from":{"first_name":"Gordo","username":"testbot"}}}},
                {"update_id":11,"callback_query":{"id":"callback"}},
                {"update_id":12,"pre_checkout_query":{"id":"checkout"}},
                {"update_id":13,"message":{"message_id":2,"successful_payment":{"telegram_payment_charge_id":"charge-1"}}},
                {"update_id":14,"edited_message":{"message_id":3}}
            ]}"#,
        );
        assert!(matches!(&actual, Ok(PollOutcome::Updates(_))));
        let updates = parsed_updates(actual);
        assert_eq!(updates.len(), 5);
        let message = message_event(&updates[0].event).ok_or("unexpected missing value")?;
        assert_eq!(
            message.message_id,
            Some(bot_core::telegram_input::MessageId(1))
        );
        assert_eq!(
            message
                .content
                .as_ref()
                .map(|content| content.text.as_str()),
            Some("hola")
        );
        assert_eq!(message.sender_last_name.as_deref(), Some("Test"));
        assert_eq!(message.chat_title.as_deref(), Some("Synthetic Group"));
        assert_eq!(
            message.replied_message_id,
            Some(bot_core::telegram_input::MessageId(3))
        );
        assert_eq!(message.replied_sender_first_name.as_deref(), Some("Gordo"));
        assert_eq!(message.replied_sender_username.as_deref(), Some("testbot"));
        assert_eq!(message.replied_text.as_deref(), Some("earlier answer"));
        assert!(message.has_reply);
        assert!(matches!(updates[1].event, IncomingEvent::CallbackQuery(_)));
        assert!(matches!(
            updates[2].event,
            IncomingEvent::PreCheckoutQuery(_)
        ));
        assert!(matches!(
            &updates[3].event,
            IncomingEvent::SuccessfulPayment(payment) if payment
                .get("successful_payment")
                .and_then(Value::as_object)
                .and_then(|payment| payment.get("telegram_payment_charge_id"))
                == Some(&json!("charge-1"))
        ));
        assert_eq!(updates[4].event, IncomingEvent::Unsupported);
        assert_eq!(next_offset(&updates, None), Some(15));
        Ok(())
    }

    #[test]
    fn message_envelope_normalizes_identity_locale_and_content_fields() -> TestResult {
        let actual = parse_response(
            200,
            r#"{"ok":true,"result":[
                {"update_id":20,"message":{
                    "message_id":"7",
                    "chat":{"id":"-42","type":"private"},
                    "from":{"id":"88","first_name":"Synthetic","username":"tester","language_code":"en-US"},
                    "reply_to_message":{"message_id":6,"voice":{"file_id":"voice-1","duration":12}},
                    "caption":"  /convertbase 101, 2, 10  ",
                    "photo":[{"file_id":"small"},{"file_id":"large"}]
                }},
                {"update_id":21,"message":{
                    "message_id":8,
                    "chat":{"id":42,"type":"group"},
                    "from":{"id":89},
                    "photo":"malformed"
                }}
            ]}"#,
        );
        let updates = parsed_updates(actual);
        let first = message_event(&updates[0].event).ok_or("unexpected missing value")?;
        assert_eq!(
            first.message_id,
            Some(bot_core::telegram_input::MessageId(7))
        );
        assert_eq!(first.chat_id, Some(bot_core::telegram_input::ChatId(-42)));
        assert_eq!(first.chat_type.as_deref(), Some("private"));
        assert_eq!(first.sender_id, Some(bot_core::telegram_input::UserId(88)));
        assert_eq!(first.sender_first_name.as_deref(), Some("Synthetic"));
        assert_eq!(first.sender_username.as_deref(), Some("tester"));
        assert!(first.has_reply);
        assert_eq!(first.sender_language_code.as_deref(), Some("en-US"));
        assert_eq!(first.visual_media_kind.as_deref(), Some("photo"));
        assert_eq!(first.audio_media_kind.as_deref(), Some("voice"));
        assert_eq!(first.audio_duration_seconds, Some(12));
        assert_eq!(
            first.content.as_ref().map(|content| (
                content.text.as_str(),
                content.photo_file_id.as_deref(),
                content.audio_file_id.as_deref()
            )),
            Some(("/convertbase 101, 2, 10", Some("large"), Some("voice-1")))
        );
        let second = message_event(&updates[1].event).ok_or("unexpected missing value")?;
        assert_eq!(second.sender_language_code, None);
        assert!(!second.has_reply);
        assert_eq!(second.content, None);
        Ok(())
    }

    #[test]
    fn captioned_media_command_keeps_direct_audio_without_a_reply() -> TestResult {
        let actual = parse_response(
            200,
            r#"{"ok":true,"result":[{"update_id":30,"message":{"message_id":9,"chat":{"id":42,"type":"private"},"from":{"id":88},"caption":" /transcribe ","voice":{"file_id":"direct-voice","duration":"4"}}}]}"#,
        );
        let updates = parsed_updates(actual);
        let message = message_event(&updates[0].event).ok_or("unexpected missing value")?;

        assert!(!message.has_reply);
        assert_eq!(message.audio_media_kind.as_deref(), Some("voice"));
        assert_eq!(message.audio_duration_seconds, Some(4));
        assert_eq!(
            message
                .content
                .as_ref()
                .map(|content| (content.text.as_str(), content.audio_file_id.as_deref())),
            Some(("/transcribe", Some("direct-voice")))
        );
        Ok(())
    }

    #[test]
    fn telegram_animation_is_exposed_as_visual_media() -> TestResult {
        let actual = parse_response(
            200,
            r#"{"ok":true,"result":[{"update_id":31,"message":{"message_id":10,"chat":{"id":42,"type":"private"},"from":{"id":88},"caption":"/transcript","animation":{"file_id":"synthetic-animation"}}}]}"#,
        );
        let updates = parsed_updates(actual);
        let message = message_event(&updates[0].event).ok_or("unexpected missing value")?;

        assert_eq!(message.visual_media_kind.as_deref(), Some("animation"));
        assert_eq!(
            message
                .content
                .as_ref()
                .and_then(|content| content.photo_file_id.as_deref()),
            Some("synthetic-animation")
        );
        Ok(())
    }

    #[test]
    fn replied_animations_preserve_file_identity_without_audio() -> TestResult {
        let result = parse_response(
            200,
            r#"{"ok":true,"result":[{"update_id":32,"message":{"message_id":11,"chat":{"id":42,"type":"private"},"from":{"id":88},"text":"/transcript","reply_to_message":{"message_id":10,"caption":"synthetic GIF","animation":{"file_id":"synthetic-animation","duration":5}}}}]}"#,
        );
        let updates = parsed_updates(result);
        let message = message_event(&updates[0].event).ok_or("unexpected missing value")?;
        assert!(message.has_reply);
        assert_eq!(message.replied_message_id, Some(MessageId(10)));
        assert_eq!(message.replied_text.as_deref(), Some("synthetic GIF"));
        assert_eq!(message.visual_media_kind.as_deref(), Some("animation"));
        assert_eq!(message.audio_media_kind, None);
        assert_eq!(message.audio_duration_seconds, None);
        assert_eq!(
            message.content.as_ref(),
            Some(&MessageContent {
                text: "/transcript".to_owned(),
                photo_file_id: Some("synthetic-animation".to_owned()),
                audio_file_id: None,
            })
        );
        Ok(())
    }

    #[test]
    fn media_duration_accepts_numbers_and_strings_and_ignores_invalid_types() -> TestResult {
        for (duration, expected) in [
            (serde_json::json!(4), Some(4)),
            (serde_json::json!(4.75), Some(4)),
            (serde_json::json!("4"), Some(4)),
            (serde_json::json!(-1), Some(0)),
            (serde_json::json!(null), None),
            (serde_json::json!({}), None),
            (serde_json::json!("invalid"), None),
        ] {
            let payload = serde_json::json!({
                "ok": true,
                "result": [{"update_id": 33, "message": {
                    "message_id": 12, "chat": {"id": 42},
                    "caption": "/transcribe", "voice": {"file_id": "synthetic-voice", "duration": duration}
                }}]
            });
            let updates = parsed_updates(parse_response(200, &payload.to_string()));
            let message = message_event(&updates[0].event).ok_or("unexpected missing value")?;
            assert_eq!(message.audio_duration_seconds, expected);
            assert_eq!(message.audio_media_kind.as_deref(), Some("voice"));
        }
        Ok(())
    }

    #[test]
    fn api_failures_preserve_conflict_rate_limit_and_status_information() {
        for error_code in [Some(401), Some(500), None] {
            assert_eq!(
                parse_response(
                    200,
                    &serde_json::json!({"ok": false, "error_code": error_code}).to_string()
                ),
                Ok(PollOutcome::Retry(PollFailure::Api { error_code }))
            );
        }
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
        let updates = parsed_updates(first);
        assert_eq!(next_offset(&updates, Some(2)), Some(9));

        let overflow = parse_response(
            200,
            &format!(r#"{{"ok":true,"result":[{{"update_id":{}}}]}}"#, i64::MAX),
        );
        assert!(matches!(&overflow, Ok(PollOutcome::Updates(_))));
        let updates = parsed_updates(overflow);
        assert_eq!(next_offset(&updates, Some(5)), Some(5));
    }
}
