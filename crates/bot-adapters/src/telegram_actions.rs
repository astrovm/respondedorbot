//! Execution of typed outbound Telegram actions.

use reqwest::Method;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use thiserror::Error;

use bot_core::telegram_actions::{ParseMode, TelegramAction, truncate_text};

use crate::telegram_http::{
    TelegramHttpError, TelegramHttpOutcome, TelegramTransport, TransportFailureKind, request_with,
};

const ACTION_TIMEOUT_SECONDS: u64 = 5;

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

fn prepare(action: TelegramAction) -> Result<PreparedAction, ActionError> {
    let (endpoint, method, params, json_payload) = match action {
        TelegramAction::SetCommands {
            commands,
            language_code,
        } => {
            let commands =
                serde_json::to_string(&commands).map_err(|_| ActionError::InvalidAction)?;
            let mut payload = Map::from_iter([("commands".to_owned(), json!(commands))]);
            insert_optional(&mut payload, "language_code", language_code)?;
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
            insert_optional(
                &mut payload,
                "reply_to_message_id",
                message.reply_to_message_id.map(|value| value.0),
            )?;
            insert_optional(
                &mut payload,
                "parse_mode",
                message.parse_mode.map(parse_mode),
            )?;
            if message.disable_web_page_preview {
                payload.insert("disable_web_page_preview".to_owned(), json!(true));
            }
            insert_optional(&mut payload, "reply_markup", message.reply_markup)?;
            (
                "sendMessage",
                Method::POST,
                None,
                Some(Value::Object(payload)),
            )
        }
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
            insert_optional(&mut payload, "text", text)?;
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
            insert_optional(&mut payload, "error_message", error_message)?;
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

fn parse_response(status_code: u16, body: &str) -> Result<ActionOutcome, ActionError> {
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
    let prepared = prepare(action)?;
    match request_with(
        transport,
        token,
        prepared.endpoint,
        prepared.method.as_str(),
        prepared.params,
        prepared.json_payload,
        ACTION_TIMEOUT_SECONDS,
    )? {
        TelegramHttpOutcome::Response { status_code, body } => parse_response(status_code, &body),
        TelegramHttpOutcome::TransportError { kind } => Ok(ActionOutcome::TransportFailed(kind)),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use bot_core::telegram_actions::{
        InlineKeyboardButton, InlineKeyboardMarkup, ParseMode, SendMessage, TelegramAction,
    };
    use bot_core::telegram_input::{ChatId, MessageId};
    use bot_core::{locale::Locale, telegram_commands::telegram_commands};

    use super::{ActionOutcome, execute_with};
    use crate::telegram_http::{
        HttpResponse, TelegramRequest, TelegramTransport, TransportFailureKind,
    };

    struct Transport {
        response: RefCell<Option<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<TelegramRequest>>,
    }

    impl TelegramTransport for Transport {
        fn send(&self, request: &TelegramRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
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
        }
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
                "reply_to_message_id":7,
                "parse_mode":"HTML",
                "disable_web_page_preview":true,
                "reply_markup":{"inline_keyboard":[[{"text":"Open","url":"https://example.test"}]]}
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
        assert_eq!(commands.as_ref().map(Vec::len), Some(68));
        assert!(
            commands.is_some_and(|commands| commands.iter().any(|command| {
                command.get("command") == Some(&serde_json::json!("help"))
                    && command.get("description") == Some(&serde_json::json!("show all commands"))
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
    }
}
