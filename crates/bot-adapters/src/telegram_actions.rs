//! Execution of typed outbound Telegram actions.

use reqwest::Method;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use thiserror::Error;

use bot_core::telegram_actions::{CommandScope, ParseMode, TelegramAction, truncate_text};
use bot_core::telegram_input::MessageId;

use crate::telegram_http::{
    TelegramHttpError, TelegramHttpOutcome, TelegramMultipartRequest, TelegramRequest,
    TelegramTransport, TransportFailureKind, send_with,
};
use std::time::Duration;

const ACTION_TIMEOUT_SECONDS: u64 = 10;
const EDIT_TIMEOUT_SECONDS: u64 = 5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionOutcome {
    Completed {
        message_id: Option<i64>,
    },
    RateLimited {
        retry_after_seconds: Option<u64>,
    },
    Failed {
        status_code: Option<u16>,
        description: String,
    },
    TransportFailed(TransportFailureKind),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ActionError {
    #[error("Telegram action could not be serialized")]
    InvalidAction,
    #[error("Telegram action response was malformed")]
    InvalidResponse,
    #[error(transparent)]
    Http(#[from] TelegramHttpError),
}

struct PreparedAction {
    endpoint: &'static str,
    method: Method,
    params: Option<Value>,
    json_payload: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct ApiEnvelope {
    ok: bool,
    #[serde(default)]
    result: Option<Value>,
    #[serde(default)]
    error_code: Option<i64>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Option<ApiParameters>,
}

#[derive(Debug, Deserialize)]
struct ApiParameters {
    #[serde(default)]
    retry_after: Option<u64>,
}

fn parse_mode(mode: ParseMode) -> &'static str {
    match mode {
        ParseMode::Html => "HTML",
        ParseMode::MarkdownV2 => "MarkdownV2",
    }
}

fn multipart_caption(caption: String, parse_mode: Option<ParseMode>) -> String {
    if parse_mode.is_some() {
        caption
    } else {
        caption.chars().take(1024).collect()
    }
}

fn insert_optional<T: serde::Serialize>(
    payload: &mut Map<String, Value>,
    field: &str,
    value: Option<T>,
) -> Result<(), ActionError> {
    if let Some(value) = value {
        payload.insert(
            field.to_owned(),
            serde_json::to_value(value).map_err(|_| ActionError::InvalidAction)?,
        );
    }
    Ok(())
}

/// Insert an optional scalar field; scalar conversions cannot fail.
fn insert_value(payload: &mut Map<String, Value>, field: &str, value: Option<impl Into<Value>>) {
    if let Some(value) = value {
        payload.insert(field.to_owned(), value.into());
    }
}

/// Replies must still be delivered when the original message was deleted
/// before the bot answered, so Telegram falls back to a plain message.
fn reply_parameters(message_id: MessageId) -> Value {
    json!({"message_id": message_id.0, "allow_sending_without_reply": true})
}

fn insert_reply(payload: &mut Map<String, Value>, reply_to_message_id: Option<MessageId>) {
    if let Some(message_id) = reply_to_message_id {
        payload.insert("reply_parameters".to_owned(), reply_parameters(message_id));
    }
}

fn push_reply_field(fields: &mut Vec<(String, String)>, reply_to_message_id: Option<MessageId>) {
    if let Some(message_id) = reply_to_message_id {
        fields.push((
            "reply_parameters".to_owned(),
            reply_parameters(message_id).to_string(),
        ));
    }
}

fn disable_link_preview(payload: &mut Map<String, Value>) {
    payload.insert(
        "link_preview_options".to_owned(),
        json!({"is_disabled": true}),
    );
}

fn prepare(action: TelegramAction) -> Result<PreparedAction, ActionError> {
    let (endpoint, method, params, json_payload) = match action {
        TelegramAction::SetCommands {
            commands,
            language_code,
            scope,
        } => {
            let commands =
                serde_json::to_string(&commands).map_err(|_| ActionError::InvalidAction)?;
            let mut payload = Map::from_iter([("commands".to_owned(), json!(commands))]);
            insert_value(&mut payload, "language_code", language_code);
            match scope {
                CommandScope::Default => {}
                CommandScope::AllGroupChats => {
                    payload.insert("scope".to_owned(), json!({"type": "all_group_chats"}));
                }
                CommandScope::Chat(chat_id) => {
                    payload.insert(
                        "scope".to_owned(),
                        json!({"type": "chat", "chat_id": chat_id.0}),
                    );
                }
            }
            (
                "setMyCommands",
                Method::POST,
                None,
                Some(Value::Object(payload)),
            )
        }
        TelegramAction::SendMessage(message) => {
            let mut payload = Map::from_iter([
                ("chat_id".to_owned(), json!(message.chat_id.0)),
                ("text".to_owned(), json!(truncate_text(&message.text))),
            ]);
            insert_reply(&mut payload, message.reply_to_message_id);
            insert_value(
                &mut payload,
                "parse_mode",
                message.parse_mode.map(parse_mode),
            );
            if message.disable_web_page_preview {
                disable_link_preview(&mut payload);
            }
            insert_optional(&mut payload, "reply_markup", message.reply_markup)?;
            (
                "sendMessage",
                Method::POST,
                None,
                Some(Value::Object(payload)),
            )
        }
        TelegramAction::SendAnimation {
            chat_id,
            animation,
            reply_to_message_id,
            caption,
        } => {
            let mut payload = Map::from_iter([
                ("chat_id".to_owned(), json!(chat_id.0)),
                ("animation".to_owned(), json!(animation)),
            ]);
            insert_reply(&mut payload, reply_to_message_id);
            insert_value(&mut payload, "caption", caption);
            (
                "sendAnimation",
                Method::POST,
                None,
                Some(Value::Object(payload)),
            )
        }
        TelegramAction::SendDocument { .. }
        | TelegramAction::SendVideo { .. }
        | TelegramAction::SendPhoto { .. }
        | TelegramAction::EditMessagePhoto { .. } => return Err(ActionError::InvalidAction),
        TelegramAction::SendInvoice {
            chat_id,
            title,
            description,
            payload,
            currency,
            prices,
        } => (
            "sendInvoice",
            Method::POST,
            None,
            Some(json!({
                "chat_id":chat_id.0,
                "title":title,
                "description":description,
                "payload":payload,
                "provider_token":"",
                "currency":currency,
                "prices":prices,
            })),
        ),
        TelegramAction::SendTyping { chat_id } => (
            "sendChatAction",
            Method::GET,
            Some(json!({"chat_id":chat_id.0,"action":"typing"})),
            None,
        ),
        TelegramAction::EditMessage {
            chat_id,
            message_id,
            text,
            reply_markup,
        } => {
            let mut payload = Map::from_iter([
                ("chat_id".to_owned(), json!(chat_id.0)),
                ("message_id".to_owned(), json!(message_id.0)),
                ("text".to_owned(), json!(truncate_text(&text))),
            ]);
            insert_optional(&mut payload, "reply_markup", reply_markup)?;
            (
                "editMessageText",
                Method::POST,
                None,
                Some(Value::Object(payload)),
            )
        }
        TelegramAction::EditMessageNoPreview {
            chat_id,
            message_id,
            text,
            reply_markup,
        } => {
            let mut payload = Map::from_iter([
                ("chat_id".to_owned(), json!(chat_id.0)),
                ("message_id".to_owned(), json!(message_id.0)),
                ("text".to_owned(), json!(truncate_text(&text))),
            ]);
            disable_link_preview(&mut payload);
            insert_optional(&mut payload, "reply_markup", reply_markup)?;
            (
                "editMessageText",
                Method::POST,
                None,
                Some(Value::Object(payload)),
            )
        }
        TelegramAction::DeleteMessage {
            chat_id,
            message_id,
        } => (
            "deleteMessage",
            Method::GET,
            Some(json!({"chat_id":chat_id.0,"message_id":message_id.0})),
            None,
        ),
        TelegramAction::AnswerCallback {
            callback_id,
            text,
            show_alert,
        } => {
            let mut payload = Map::from_iter([
                ("callback_query_id".to_owned(), json!(callback_id)),
                ("show_alert".to_owned(), json!(show_alert)),
            ]);
            insert_value(&mut payload, "text", text);
            (
                "answerCallbackQuery",
                Method::POST,
                None,
                Some(Value::Object(payload)),
            )
        }
        TelegramAction::AnswerPreCheckout {
            query_id,
            ok,
            error_message,
        } => {
            let mut payload = Map::from_iter([
                ("pre_checkout_query_id".to_owned(), json!(query_id)),
                ("ok".to_owned(), json!(ok)),
            ]);
            insert_value(&mut payload, "error_message", error_message);
            (
                "answerPreCheckoutQuery",
                Method::POST,
                None,
                Some(Value::Object(payload)),
            )
        }
    };
    Ok(PreparedAction {
        endpoint,
        method,
        params,
        json_payload,
    })
}

fn parse_response(
    status_code: u16,
    body: &str,
    edited_message_id: Option<i64>,
) -> Result<ActionOutcome, ActionError> {
    let envelope = serde_json::from_str::<ApiEnvelope>(body);
    if status_code == 429 {
        return Ok(ActionOutcome::RateLimited {
            retry_after_seconds: envelope
                .ok()
                .and_then(|value| value.parameters)
                .and_then(|value| value.retry_after),
        });
    }
    let envelope = match envelope {
        Ok(envelope) => envelope,
        Err(_) if !(200..300).contains(&status_code) => {
            return Ok(ActionOutcome::Failed {
                status_code: Some(status_code),
                description: format!("Telegram HTTP {status_code}"),
            });
        }
        Err(_) => return Err(ActionError::InvalidResponse),
    };
    if envelope.ok && (200..300).contains(&status_code) {
        let message_id = envelope
            .result
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|result| result.get("message_id"))
            .and_then(Value::as_i64);
        return Ok(ActionOutcome::Completed { message_id });
    }
    if envelope.error_code == Some(429) {
        return Ok(ActionOutcome::RateLimited {
            retry_after_seconds: envelope.parameters.and_then(|value| value.retry_after),
        });
    }
    // A timed-out edit may already have reached Telegram. Its retry is a
    // success when the requested content is already present.
    if status_code == 400
        && edited_message_id.is_some()
        && envelope.description.as_deref().is_some_and(|description| {
            description.starts_with("Bad Request: message is not modified")
        })
    {
        return Ok(ActionOutcome::Completed {
            message_id: edited_message_id,
        });
    }
    Ok(ActionOutcome::Failed {
        status_code: Some(status_code),
        description: envelope
            .description
            .unwrap_or_else(|| "telegram request failed".to_owned()),
    })
}

pub fn execute_with<T: TelegramTransport>(
    transport: &T,
    token: &str,
    action: TelegramAction,
) -> Result<ActionOutcome, ActionError> {
    let edited_message_id = match &action {
        TelegramAction::EditMessage { message_id, .. }
        | TelegramAction::EditMessageNoPreview { message_id, .. }
        | TelegramAction::EditMessagePhoto { message_id, .. } => Some(message_id.0),
        _ => None,
    };
    let request = match action {
        TelegramAction::SendDocument {
            chat_id,
            document,
            file_name,
            reply_to_message_id,
            caption,
        } => {
            let mut fields = vec![
                ("chat_id".to_owned(), chat_id.0.to_string()),
                (
                    "caption".to_owned(),
                    caption.chars().take(1024).collect::<String>(),
                ),
            ];
            push_reply_field(&mut fields, reply_to_message_id);
            TelegramMultipartRequest {
                token: token.to_owned(),
                endpoint: "sendDocument".to_owned(),
                fields,
                file_field: "document".to_owned(),
                file_name,
                file_bytes: document,
                content_type: "text/plain; charset=utf-8".to_owned(),
                timeout: Duration::from_secs(60),
            }
        }
        TelegramAction::SendVideo {
            chat_id,
            video,
            reply_to_message_id,
            caption,
            reply_markup,
        } => {
            let mut fields = vec![
                ("chat_id".to_owned(), chat_id.0.to_string()),
                (
                    "caption".to_owned(),
                    caption.chars().take(1024).collect::<String>(),
                ),
                ("supports_streaming".to_owned(), "true".to_owned()),
            ];
            push_reply_field(&mut fields, reply_to_message_id);
            if let Some(reply_markup) = reply_markup {
                fields.push((
                    "reply_markup".to_owned(),
                    serde_json::to_string(&reply_markup).map_err(|_| ActionError::InvalidAction)?,
                ));
            }
            TelegramMultipartRequest {
                token: token.to_owned(),
                endpoint: "sendVideo".to_owned(),
                fields,
                file_field: "video".to_owned(),
                file_name: "instagram.mp4".to_owned(),
                file_bytes: video,
                content_type: "video/mp4".to_owned(),
                timeout: Duration::from_secs(60),
            }
        }
        TelegramAction::SendPhoto {
            chat_id,
            photo,
            reply_to_message_id,
            caption,
            parse_mode: caption_parse_mode,
            reply_markup,
        } => {
            let mut fields = vec![
                ("chat_id".to_owned(), chat_id.0.to_string()),
                (
                    "caption".to_owned(),
                    multipart_caption(caption, caption_parse_mode),
                ),
            ];
            push_reply_field(&mut fields, reply_to_message_id);
            if let Some(mode) = caption_parse_mode {
                fields.push(("parse_mode".to_owned(), parse_mode(mode).to_owned()));
            }
            if let Some(reply_markup) = reply_markup {
                fields.push((
                    "reply_markup".to_owned(),
                    serde_json::to_string(&reply_markup).map_err(|_| ActionError::InvalidAction)?,
                ));
            }
            TelegramMultipartRequest {
                token: token.to_owned(),
                endpoint: "sendPhoto".to_owned(),
                fields,
                file_field: "photo".to_owned(),
                file_name: "signal.png".to_owned(),
                file_bytes: photo,
                content_type: "image/png".to_owned(),
                timeout: Duration::from_secs(60),
            }
        }
        TelegramAction::EditMessagePhoto {
            chat_id,
            message_id,
            photo,
            caption,
            parse_mode: caption_parse_mode,
            reply_markup,
        } => {
            let mut media = Map::from_iter([
                ("type".to_owned(), json!("photo")),
                ("media".to_owned(), json!("attach://photo")),
                (
                    "caption".to_owned(),
                    json!(multipart_caption(caption, caption_parse_mode)),
                ),
            ]);
            if let Some(mode) = caption_parse_mode {
                media.insert("parse_mode".to_owned(), json!(parse_mode(mode)));
            }
            let mut fields = vec![
                ("chat_id".to_owned(), chat_id.0.to_string()),
                ("message_id".to_owned(), message_id.0.to_string()),
                (
                    "media".to_owned(),
                    serde_json::to_string(&Value::Object(media))
                        .map_err(|_| ActionError::InvalidAction)?,
                ),
            ];
            if let Some(reply_markup) = reply_markup {
                fields.push((
                    "reply_markup".to_owned(),
                    serde_json::to_string(&reply_markup).map_err(|_| ActionError::InvalidAction)?,
                ));
            }
            TelegramMultipartRequest {
                token: token.to_owned(),
                endpoint: "editMessageMedia".to_owned(),
                fields,
                file_field: "photo".to_owned(),
                file_name: "signal.png".to_owned(),
                file_bytes: photo,
                content_type: "image/png".to_owned(),
                timeout: Duration::from_secs(60),
            }
        }
        action => {
            let prepared = prepare(action)?;
            let request = TelegramRequest {
                token: token.to_owned(),
                endpoint: prepared.endpoint.to_owned(),
                method: prepared.method,
                params: prepared.params,
                json_payload: prepared.json_payload,
                timeout: Duration::from_secs(if edited_message_id.is_some() {
                    EDIT_TIMEOUT_SECONDS
                } else {
                    ACTION_TIMEOUT_SECONDS
                }),
            };
            return match send_with(transport, &request) {
                TelegramHttpOutcome::Response { status_code, body } => {
                    parse_response(status_code, &body, edited_message_id)
                }
                TelegramHttpOutcome::TransportError { kind } => {
                    Ok(ActionOutcome::TransportFailed(kind))
                }
            };
        }
    };
    match transport.send_action_multipart(&request) {
        Ok(response) => parse_response(response.status_code, &response.body, edited_message_id),
        Err(kind) => Ok(ActionOutcome::TransportFailed(kind)),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::time::Duration;

    use bot_core::telegram_actions::{
        CommandScope, InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice, ParseMode,
        SendMessage, TelegramAction,
    };
    use bot_core::telegram_input::{ChatId, MessageId};
    use bot_core::{locale::Locale, telegram_commands::telegram_commands};

    use super::{ActionError, ActionOutcome, execute_with, multipart_caption, prepare};
    use crate::telegram_http::{
        HttpResponse, TelegramMultipartRequest, TelegramRequest, TelegramTransport,
        TransportFailureKind,
    };

    const REPLY_TO_7: &str = r#"{"allow_sending_without_reply":true,"message_id":7}"#;

    #[test]
    fn unchanged_edits_are_successful_without_hiding_other_rejections() {
        let unchanged = r#"{"ok":false,"error_code":400,"description":"Bad Request: message is not modified: specified new message content and reply markup are exactly the same"}"#;
        for action in [
            TelegramAction::EditMessage {
                chat_id: ChatId(-42),
                message_id: MessageId(7),
                text: "final".to_owned(),
                reply_markup: None,
            },
            TelegramAction::EditMessageNoPreview {
                chat_id: ChatId(-42),
                message_id: MessageId(7),
                text: "final".to_owned(),
                reply_markup: None,
            },
            TelegramAction::EditMessagePhoto {
                chat_id: ChatId(-42),
                message_id: MessageId(7),
                photo: Vec::new().into(),
                caption: "final".to_owned(),
                parse_mode: None,
                reply_markup: None,
            },
        ] {
            assert_eq!(
                execute_with(&transport_with_status(400, unchanged), "synthetic", action),
                Ok(ActionOutcome::Completed {
                    message_id: Some(7)
                })
            );
        }
        for (status, body, action) in [
            (
                400,
                unchanged,
                TelegramAction::SendMessage(SendMessage::new(ChatId(-42), "final")),
            ),
            (
                500,
                unchanged,
                TelegramAction::EditMessage {
                    chat_id: ChatId(-42),
                    message_id: MessageId(7),
                    text: "final".to_owned(),
                    reply_markup: None,
                },
            ),
            (
                400,
                r#"{"ok":false,"description":"Bad Request: message to edit not found"}"#,
                TelegramAction::EditMessage {
                    chat_id: ChatId(-42),
                    message_id: MessageId(7),
                    text: "final".to_owned(),
                    reply_markup: None,
                },
            ),
        ] {
            assert!(matches!(
                execute_with(&transport_with_status(status, body), "synthetic", action),
                Ok(ActionOutcome::Failed { .. })
            ));
        }
    }

    struct Transport {
        response: RefCell<Option<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<TelegramRequest>>,
        multipart_requests: RefCell<Vec<TelegramMultipartRequest>>,
    }

    impl TelegramTransport for Transport {
        fn send(&self, request: &TelegramRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            self.response
                .borrow_mut()
                .take()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
        fn send_action_multipart(
            &self,
            request: &TelegramMultipartRequest,
        ) -> Result<HttpResponse, TransportFailureKind> {
            self.multipart_requests.borrow_mut().push(request.clone());
            self.response
                .borrow_mut()
                .take()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    fn transport(body: &str) -> Transport {
        transport_with_status(200, body)
    }

    fn transport_with_status(status_code: u16, body: &str) -> Transport {
        Transport {
            response: RefCell::new(Some(Ok(HttpResponse {
                status_code,
                body: body.to_owned(),
            }))),
            requests: RefCell::new(Vec::new()),
            multipart_requests: RefCell::new(Vec::new()),
        }
    }

    #[test]
    fn video_upload_uses_bounded_multipart_contract_and_returns_message_id() {
        let transport = transport(r#"{"ok":true,"result":{"message_id":44}}"#);
        let markup = InlineKeyboardMarkup {
            inline_keyboard: vec![vec![InlineKeyboardButton {
                text: "Original".to_owned(),
                url: Some("https://instagram.com/reel/a".to_owned()),
                callback_data: None,
                copy_text: None,
            }]],
        };
        assert_eq!(
            execute_with(
                &transport,
                "synthetic-token",
                TelegramAction::SendVideo {
                    chat_id: ChatId(42),
                    video: vec![1, 2, 3].into(),
                    reply_to_message_id: Some(MessageId(7)),
                    caption: "fixed".to_owned(),
                    reply_markup: Some(markup),
                },
            ),
            Ok(ActionOutcome::Completed {
                message_id: Some(44)
            })
        );
        assert_eq!(
            execute_with(
                &transport,
                "synthetic-token",
                TelegramAction::SendTyping {
                    chat_id: ChatId(42)
                },
            ),
            Ok(ActionOutcome::TransportFailed(
                TransportFailureKind::Request
            ))
        );
        assert_eq!(transport.requests.borrow().len(), 1);
        let requests = transport.multipart_requests.borrow();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].endpoint, "sendVideo");
        assert_eq!(requests[0].file_field, "video");
        assert_eq!(requests[0].file_name, "instagram.mp4");
        assert_eq!(requests[0].file_bytes.as_ref(), [1, 2, 3]);
        assert_eq!(requests[0].content_type, "video/mp4");
        assert!(
            requests[0]
                .fields
                .contains(&("chat_id".to_owned(), "42".to_owned()))
        );
        assert!(
            requests[0]
                .fields
                .contains(&("reply_parameters".to_owned(), REPLY_TO_7.to_owned()))
        );
        assert!(
            requests[0]
                .fields
                .contains(&("supports_streaming".to_owned(), "true".to_owned()))
        );
        assert_eq!(requests[0].timeout, Duration::from_secs(60));
    }

    #[test]
    fn document_upload_preserves_complete_utf8_text() {
        let transport = transport(r#"{"ok":true,"result":{"message_id":45}}"#);
        let text = "texto sintético 🦀".repeat(300);
        assert_eq!(
            execute_with(
                &transport,
                "synthetic-token",
                TelegramAction::SendDocument {
                    chat_id: ChatId(42),
                    document: text.as_bytes().to_vec().into(),
                    file_name: "transcript.txt".to_owned(),
                    reply_to_message_id: Some(MessageId(7)),
                    caption: "full transcript".to_owned(),
                },
            ),
            Ok(ActionOutcome::Completed {
                message_id: Some(45)
            })
        );
        let requests = transport.multipart_requests.borrow();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].endpoint, "sendDocument");
        assert_eq!(requests[0].file_field, "document");
        assert_eq!(requests[0].file_name, "transcript.txt");
        assert_eq!(requests[0].file_bytes.as_ref(), text.as_bytes());
        assert_eq!(requests[0].timeout, Duration::from_secs(60));
        assert_eq!(requests[0].token, "synthetic-token");
        assert_eq!(requests[0].fields.len(), 3);
        assert!(
            requests[0]
                .fields
                .contains(&("chat_id".to_owned(), "42".to_owned()))
        );
        assert_eq!(requests[0].content_type, "text/plain; charset=utf-8");
        assert!(
            requests[0]
                .fields
                .contains(&("caption".to_owned(), "full transcript".to_owned()))
        );
        assert!(
            requests[0]
                .fields
                .contains(&("reply_parameters".to_owned(), REPLY_TO_7.to_owned()))
        );
    }

    #[test]
    fn document_upload_preserves_delivery_failures_and_bounds_caption() {
        for (status, body, expected) in [
            (
                429,
                r#"{"ok":false,"parameters":{"retry_after":7}}"#,
                Ok(ActionOutcome::RateLimited {
                    retry_after_seconds: Some(7),
                }),
            ),
            (
                429,
                "not json",
                Ok(ActionOutcome::RateLimited {
                    retry_after_seconds: None,
                }),
            ),
            (
                403,
                r#"{"ok":false,"description":"bot blocked"}"#,
                Ok(ActionOutcome::Failed {
                    status_code: Some(403),
                    description: "bot blocked".to_owned(),
                }),
            ),
            (200, "not json", Err(ActionError::InvalidResponse)),
            (
                200,
                r#"{"ok":true,"result":true}"#,
                Ok(ActionOutcome::Completed { message_id: None }),
            ),
        ] {
            let transport = transport_with_status(status, body);
            let action = TelegramAction::SendDocument {
                chat_id: ChatId(42),
                document: b"complete synthetic transcript".to_vec().into(),
                file_name: "transcript.txt".to_owned(),
                reply_to_message_id: None,
                caption: "🦀".repeat(1100),
            };
            assert_eq!(
                execute_with(&transport, "synthetic-token", action.clone()),
                expected
            );
            let requests = transport.multipart_requests.borrow();
            assert_eq!(requests.len(), 1);
            assert!(
                !requests[0]
                    .fields
                    .iter()
                    .any(|(key, _)| key == "reply_parameters")
            );
            assert!(
                requests[0]
                    .fields
                    .contains(&("caption".to_owned(), "🦀".repeat(1024)))
            );
            assert_eq!(
                requests[0].file_bytes.as_ref(),
                b"complete synthetic transcript"
            );
            drop(requests);
            assert_eq!(
                execute_with(&transport, "synthetic-token", action),
                Ok(ActionOutcome::TransportFailed(
                    TransportFailureKind::Request
                ))
            );
        }
    }

    #[test]
    fn photo_send_and_refresh_use_png_multipart_and_typed_media() {
        let sent = r#"{"ok":true,"result":{"message_id":55}}"#;
        let transport = transport(sent);
        let markup = InlineKeyboardMarkup {
            inline_keyboard: vec![vec![InlineKeyboardButton {
                text: "Details".to_owned(),
                url: Some("https://example.test/details".to_owned()),
                callback_data: None,
                copy_text: None,
            }]],
        };
        assert_eq!(
            execute_with(
                &transport,
                "synthetic-token",
                TelegramAction::SendPhoto {
                    chat_id: ChatId(42),
                    photo: vec![1, 2, 3].into(),
                    reply_to_message_id: Some(MessageId(7)),
                    caption: "<b>signal</b>".to_owned(),
                    parse_mode: Some(ParseMode::MarkdownV2),
                    reply_markup: Some(markup.clone()),
                },
            ),
            Ok(ActionOutcome::Completed {
                message_id: Some(55)
            })
        );
        transport.response.replace(Some(Ok(HttpResponse {
            status_code: 200,
            body: sent.to_owned(),
        })));
        assert_eq!(
            execute_with(
                &transport,
                "synthetic-token",
                TelegramAction::EditMessagePhoto {
                    chat_id: ChatId(42),
                    message_id: MessageId(55),
                    photo: vec![4, 5, 6].into(),
                    caption: "<b>updated</b>".to_owned(),
                    parse_mode: Some(ParseMode::Html),
                    reply_markup: Some(markup),
                },
            ),
            Ok(ActionOutcome::Completed {
                message_id: Some(55)
            })
        );
        assert_eq!(
            execute_with(
                &transport,
                "synthetic-token",
                TelegramAction::SendTyping {
                    chat_id: ChatId(42)
                },
            ),
            Ok(ActionOutcome::TransportFailed(
                TransportFailureKind::Request
            ))
        );
        assert_eq!(transport.requests.borrow().len(), 1);
        let requests = transport.multipart_requests.borrow();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].endpoint, "sendPhoto");
        assert_eq!(requests[0].file_field, "photo");
        assert_eq!(requests[0].file_bytes.as_ref(), [1, 2, 3]);
        assert!(
            requests[0]
                .fields
                .contains(&("parse_mode".to_owned(), "MarkdownV2".to_owned()))
        );
        assert!(requests[0].fields.iter().any(|(key, value)| {
            key == "reply_markup"
                && value.contains("Details")
                && value.contains("https://example.test/details")
        }));
        assert_eq!(requests[1].endpoint, "editMessageMedia");
        let media = requests[1]
            .fields
            .iter()
            .find(|(key, _value)| key == "media")
            .map(|(_key, value)| value.as_str());
        assert!(media.is_some_and(|media| {
            media.contains("attach://photo")
                && media.contains("<b>updated</b>")
                && media.contains("HTML")
        }));
        assert!(requests[1].fields.iter().any(|(key, value)| {
            key == "reply_markup"
                && value.contains("Details")
                && value.contains("https://example.test/details")
        }));
    }

    #[test]
    fn multipart_transport_failures_remain_typed() {
        let transport = transport("");
        transport
            .response
            .replace(Some(Err(TransportFailureKind::Request)));

        assert_eq!(
            execute_with(
                &transport,
                "synthetic-token",
                TelegramAction::SendPhoto {
                    chat_id: ChatId(42),
                    photo: vec![1, 2, 3].into(),
                    reply_to_message_id: None,
                    caption: "synthetic caption".to_owned(),
                    parse_mode: None,
                    reply_markup: None,
                },
            ),
            Ok(ActionOutcome::TransportFailed(
                TransportFailureKind::Request
            ))
        );
    }

    #[test]
    fn html_photo_captions_are_not_truncated_inside_markup() {
        let caption = format!(
            "<a href=\"https://example.test/{}\">link</a>",
            "x".repeat(1_024)
        );
        assert!(caption.len() > 1_024);
        assert_eq!(
            multipart_caption(caption.clone(), Some(ParseMode::Html)),
            caption
        );
        assert_eq!(multipart_caption("x".repeat(1_100), None).len(), 1_024);
    }

    #[test]
    fn send_message_preserves_typed_options_and_returns_message_id() {
        let transport = transport(r#"{"ok":true,"result":{"message_id":77}}"#);
        let action = TelegramAction::SendMessage(SendMessage {
            chat_id: ChatId(-10042),
            text: "hello".to_owned(),
            reply_to_message_id: Some(MessageId(7)),
            parse_mode: Some(ParseMode::Html),
            disable_web_page_preview: true,
            reply_markup: Some(InlineKeyboardMarkup {
                inline_keyboard: vec![vec![InlineKeyboardButton {
                    text: "Open".to_owned(),
                    url: Some("https://example.test".to_owned()),
                    callback_data: None,
                    copy_text: None,
                }]],
            }),
        });
        assert_eq!(
            execute_with(&transport, "synthetic-token", action),
            Ok(ActionOutcome::Completed {
                message_id: Some(77)
            })
        );
        let requests = transport.requests.borrow();
        assert_eq!(requests[0].endpoint, "sendMessage");
        assert_eq!(
            requests[0].json_payload,
            Some(serde_json::json!({
                "chat_id":-10042,
                "text":"hello",
                "reply_parameters":{"message_id":7,"allow_sending_without_reply":true},
                "parse_mode":"HTML",
                "link_preview_options":{"is_disabled":true},
                "reply_markup":{"inline_keyboard":[[{"text":"Open","url":"https://example.test"}]]}
            }))
        );
    }

    #[test]
    fn draft_edit_disables_web_page_previews() {
        let transport = transport(r#"{"ok":true,"result":{"message_id":77}}"#);
        assert_eq!(
            execute_with(
                &transport,
                "synthetic-token",
                TelegramAction::EditMessageNoPreview {
                    chat_id: ChatId(42),
                    message_id: MessageId(77),
                    text: "draft https://example.test".to_owned(),
                    reply_markup: None,
                },
            ),
            Ok(ActionOutcome::Completed {
                message_id: Some(77)
            })
        );
        let request = &transport.requests.borrow()[0];
        assert_eq!(request.endpoint, "editMessageText");
        assert_eq!(
            request.json_payload,
            Some(serde_json::json!({
                "chat_id":42,
                "message_id":77,
                "text":"draft https://example.test",
                "link_preview_options":{"is_disabled":true},
            }))
        );
    }

    #[test]
    fn set_commands_preserves_legacy_serialized_menu_payload() {
        let transport = transport(r#"{"ok":true,"result":true}"#);
        assert_eq!(
            execute_with(
                &transport,
                "synthetic-token",
                TelegramAction::SetCommands {
                    commands: telegram_commands(Locale::En),
                    language_code: Some("en".to_owned()),
                    scope: CommandScope::Default,
                },
            ),
            Ok(ActionOutcome::Completed { message_id: None })
        );
        let requests = transport.requests.borrow();
        assert_eq!(requests[0].endpoint, "setMyCommands");
        let payload = requests[0]
            .json_payload
            .as_ref()
            .and_then(serde_json::Value::as_object);
        assert_eq!(
            payload
                .and_then(|value| value.get("language_code"))
                .and_then(serde_json::Value::as_str),
            Some("en")
        );
        let commands = payload
            .and_then(|value| value.get("commands"))
            .and_then(serde_json::Value::as_str)
            .and_then(|value| serde_json::from_str::<Vec<serde_json::Value>>(value).ok());
        assert_eq!(commands.as_ref().map(Vec::len), Some(86));
        assert!(
            commands.is_some_and(|commands| commands.iter().any(|command| {
                command.get("command") == Some(&serde_json::json!("help"))
                    && command.get("description")
                        == Some(&serde_json::json!("help and command list"))
            }))
        );
    }

    #[test]
    fn set_commands_sends_group_and_chat_scopes() {
        for (scope, expected) in [
            (CommandScope::Default, None),
            (
                CommandScope::AllGroupChats,
                Some(serde_json::json!({"type": "all_group_chats"})),
            ),
            (
                CommandScope::Chat(ChatId(-42)),
                Some(serde_json::json!({"type": "chat", "chat_id": -42})),
            ),
        ] {
            let transport = transport(r#"{"ok":true,"result":true}"#);
            assert_eq!(
                execute_with(
                    &transport,
                    "synthetic-token",
                    TelegramAction::SetCommands {
                        commands: telegram_commands(Locale::Es),
                        language_code: None,
                        scope,
                    },
                ),
                Ok(ActionOutcome::Completed { message_id: None })
            );
            let requests = transport.requests.borrow();
            let payload = requests[0]
                .json_payload
                .as_ref()
                .and_then(serde_json::Value::as_object);
            assert_eq!(
                payload.and_then(|value| value.get("scope")).cloned(),
                expected
            );
            assert!(payload.is_some_and(|value| !value.contains_key("language_code")));
        }
    }

    #[test]
    fn send_invoice_preserves_stars_payload_and_empty_provider_token() {
        let transport = transport(r#"{"ok":true,"result":{"message_id":9}}"#);
        assert_eq!(
            execute_with(
                &transport,
                "synthetic-token",
                TelegramAction::SendInvoice {
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
            ),
            Ok(ActionOutcome::Completed {
                message_id: Some(9)
            })
        );
        let request = &transport.requests.borrow()[0];
        assert_eq!(request.endpoint, "sendInvoice");
        assert_eq!(
            request.json_payload,
            Some(serde_json::json!({
                "chat_id":42,
                "title":"50.00 AI credit pack",
                "description":"Add 50.00 credits for AI messages",
                "payload":"topup:p50:42:en",
                "provider_token":"",
                "currency":"XTR",
                "prices":[{"label":"50.00 AI credits","amount":25}],
            }))
        );
    }

    #[test]
    fn send_animation_preserves_url_reply_and_optional_caption() {
        let transport = transport(r#"{"ok":true,"result":{"message_id":10}}"#);
        assert_eq!(
            execute_with(
                &transport,
                "synthetic-token",
                TelegramAction::SendAnimation {
                    chat_id: ChatId(42),
                    animation: "https://example.test/greeting.gif".to_owned(),
                    reply_to_message_id: Some(MessageId(7)),
                    caption: Some("hello".to_owned()),
                },
            ),
            Ok(ActionOutcome::Completed {
                message_id: Some(10)
            })
        );
        let request = &transport.requests.borrow()[0];
        assert_eq!(request.endpoint, "sendAnimation");
        assert_eq!(
            request.json_payload,
            Some(serde_json::json!({
                "chat_id":42,
                "animation":"https://example.test/greeting.gif",
                "reply_parameters":{"message_id":7,"allow_sending_without_reply":true},
                "caption":"hello",
            }))
        );
    }

    #[test]
    fn plans_typing_edit_delete_callback_and_checkout_endpoints() {
        let actions = [
            (
                TelegramAction::SendTyping { chat_id: ChatId(1) },
                "sendChatAction",
            ),
            (
                TelegramAction::EditMessage {
                    chat_id: ChatId(1),
                    message_id: MessageId(2),
                    text: "edit".to_owned(),
                    reply_markup: None,
                },
                "editMessageText",
            ),
            (
                TelegramAction::DeleteMessage {
                    chat_id: ChatId(1),
                    message_id: MessageId(2),
                },
                "deleteMessage",
            ),
            (
                TelegramAction::AnswerCallback {
                    callback_id: "callback".to_owned(),
                    text: Some("done".to_owned()),
                    show_alert: true,
                },
                "answerCallbackQuery",
            ),
            (
                TelegramAction::AnswerPreCheckout {
                    query_id: "checkout".to_owned(),
                    ok: false,
                    error_message: Some("invalid".to_owned()),
                },
                "answerPreCheckoutQuery",
            ),
        ];
        for (action, endpoint) in actions {
            let transport = transport(r#"{"ok":true,"result":true}"#);
            assert_eq!(
                execute_with(&transport, "token", action),
                Ok(ActionOutcome::Completed { message_id: None })
            );
            assert_eq!(transport.requests.borrow()[0].endpoint, endpoint);
        }
    }

    #[test]
    fn classifies_rate_limits_api_failures_malformed_and_transport_errors() {
        let rate_limit = transport(
            r#"{"ok":false,"error_code":429,"description":"slow down","parameters":{"retry_after":2}}"#,
        );
        assert_eq!(
            execute_with(
                &rate_limit,
                "token",
                TelegramAction::SendTyping { chat_id: ChatId(1) }
            ),
            Ok(ActionOutcome::RateLimited {
                retry_after_seconds: Some(2)
            })
        );

        let failed = transport(r#"{"ok":false,"error_code":400,"description":"bad request"}"#);
        assert_eq!(
            execute_with(
                &failed,
                "token",
                TelegramAction::SendTyping { chat_id: ChatId(1) }
            ),
            Ok(ActionOutcome::Failed {
                status_code: Some(200),
                description: "bad request".to_owned()
            })
        );

        let malformed = transport("not-json");
        assert!(
            execute_with(
                &malformed,
                "token",
                TelegramAction::SendTyping { chat_id: ChatId(1) }
            )
            .is_err()
        );

        let http_failure = transport_with_status(503, "upstream unavailable");
        assert_eq!(
            execute_with(
                &http_failure,
                "token",
                TelegramAction::SendTyping { chat_id: ChatId(1) }
            ),
            Ok(ActionOutcome::Failed {
                status_code: Some(503),
                description: "Telegram HTTP 503".to_owned()
            })
        );

        let transport_error = Transport {
            response: RefCell::new(Some(Err(TransportFailureKind::Timeout))),
            requests: RefCell::new(Vec::new()),
            multipart_requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            execute_with(
                &transport_error,
                "token",
                TelegramAction::SendTyping { chat_id: ChatId(1) }
            ),
            Ok(ActionOutcome::TransportFailed(
                TransportFailureKind::Timeout
            ))
        );

        assert!(matches!(
            prepare(TelegramAction::SendVideo {
                chat_id: ChatId(1),
                video: vec![1].into(),
                reply_to_message_id: None,
                caption: String::new(),
                reply_markup: None,
            }),
            Err(ActionError::InvalidAction)
        ));
    }

    #[test]
    fn failures_without_a_description_use_a_generic_message() {
        let transport = transport_with_status(400, r#"{"ok":false}"#);
        assert_eq!(
            execute_with(
                &transport,
                "synthetic-token",
                TelegramAction::SendTyping {
                    chat_id: ChatId(42)
                },
            ),
            Ok(ActionOutcome::Failed {
                status_code: Some(400),
                description: "telegram request failed".to_owned(),
            })
        );
        assert_eq!(transport.requests.borrow()[0].endpoint, "sendChatAction");
    }

    #[test]
    fn media_uploads_without_keyboards_omit_reply_markup() {
        let sent = r#"{"ok":true,"result":{"message_id":56}}"#;
        let transport = transport(sent);
        assert_eq!(
            execute_with(
                &transport,
                "synthetic-token",
                TelegramAction::SendVideo {
                    chat_id: ChatId(42),
                    video: vec![1].into(),
                    reply_to_message_id: None,
                    caption: "plain".to_owned(),
                    reply_markup: None,
                },
            ),
            Ok(ActionOutcome::Completed {
                message_id: Some(56)
            })
        );
        transport.response.replace(Some(Ok(HttpResponse {
            status_code: 200,
            body: sent.to_owned(),
        })));
        assert_eq!(
            execute_with(
                &transport,
                "synthetic-token",
                TelegramAction::EditMessagePhoto {
                    chat_id: ChatId(42),
                    message_id: MessageId(56),
                    photo: vec![2].into(),
                    caption: "plain".to_owned(),
                    parse_mode: None,
                    reply_markup: None,
                },
            ),
            Ok(ActionOutcome::Completed {
                message_id: Some(56)
            })
        );
        let requests = transport.multipart_requests.borrow();
        assert_eq!(requests.len(), 2);
        assert!(
            requests
                .iter()
                .all(|request| !request.fields.iter().any(|(key, _)| key == "reply_markup"))
        );
        assert!(
            !requests[0]
                .fields
                .iter()
                .any(|(key, _)| key == "reply_parameters")
        );
    }
}
