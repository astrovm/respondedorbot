//! Typed parsing and classification for Telegram callback queries.

use serde::Serialize;
use serde_json::{Map, Value};
use thiserror::Error;

use crate::telegram_input::{python_string, python_truthy};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CallbackRoute {
    Topup,
    Charges,
    Task,
    Signal,
    Config,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CallbackContext {
    pub callback_id: Option<String>,
    pub data: String,
    pub chat_id: String,
    pub chat_type: String,
    pub message_id: i64,
    pub user_id: Option<i64>,
    pub user_language_code: Option<String>,
    pub route: CallbackRoute,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CallbackContextOutcome {
    Guard,
    Context { context: CallbackContext },
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CallbackParseError {
    #[error("Telegram callback query must be an object")]
    InvalidCallback,
    #[error("Telegram callback message, chat, or sender is malformed")]
    InvalidNestedObject,
    #[error("Telegram callback message id is invalid")]
    InvalidMessageId,
}

fn object_or_empty(value: Option<&Value>) -> Result<&Map<String, Value>, CallbackParseError> {
    match value {
        Some(Value::Object(value)) => Ok(value),
        Some(value) if python_truthy(value) => Err(CallbackParseError::InvalidNestedObject),
        Some(_) | None => Ok(empty_object()),
    }
}

fn empty_object() -> &'static Map<String, Value> {
    static EMPTY: std::sync::OnceLock<Map<String, Value>> = std::sync::OnceLock::new();
    EMPTY.get_or_init(Map::new)
}

fn optional_truthy_string(value: Option<&Value>) -> Option<String> {
    value
        .filter(|value| python_truthy(value))
        .map(python_string)
}

fn strict_python_int(value: &Value) -> Option<i64> {
    python_string(value).parse().ok()
}

#[must_use]
pub fn classify_callback(data: &str) -> CallbackRoute {
    if data.starts_with("topup:") {
        CallbackRoute::Topup
    } else if data.starts_with("chg:") {
        CallbackRoute::Charges
    } else if data.starts_with("task:") {
        CallbackRoute::Task
    } else if data.starts_with("sig:") {
        CallbackRoute::Signal
    } else if data.starts_with("cfg:") {
        CallbackRoute::Config
    } else {
        CallbackRoute::Unknown
    }
}

pub fn parse_callback_context(
    callback_query: &Value,
) -> Result<CallbackContextOutcome, CallbackParseError> {
    let callback_query = callback_query
        .as_object()
        .ok_or(CallbackParseError::InvalidCallback)?;
    let callback_id = optional_truthy_string(callback_query.get("id"));
    let Some(data) = optional_truthy_string(callback_query.get("data")) else {
        return Ok(CallbackContextOutcome::Guard);
    };
    let message = object_or_empty(callback_query.get("message"))?;
    let chat = object_or_empty(message.get("chat"))?;
    let Some(chat_id_value) = chat.get("id").filter(|value| !value.is_null()) else {
        return Ok(CallbackContextOutcome::Guard);
    };
    let Some(message_id_value) = message.get("message_id").filter(|value| !value.is_null()) else {
        return Ok(CallbackContextOutcome::Guard);
    };
    let message_id =
        strict_python_int(message_id_value).ok_or(CallbackParseError::InvalidMessageId)?;
    let user = object_or_empty(callback_query.get("from"))?;
    let user_id = user.get("id").and_then(strict_python_int);
    let user_language_code = optional_truthy_string(user.get("language_code"));
    Ok(CallbackContextOutcome::Context {
        context: CallbackContext {
            callback_id,
            route: classify_callback(&data),
            data,
            chat_id: python_string(chat_id_value),
            chat_type: chat.get("type").map_or_else(String::new, python_string),
            message_id,
            user_id,
            user_language_code,
        },
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        CallbackContext, CallbackContextOutcome, CallbackParseError, CallbackRoute,
        classify_callback, parse_callback_context,
    };

    #[test]
    fn parses_complete_callback_and_classifies_every_route() {
        assert_eq!(
            parse_callback_context(&json!({
                "id":"callback-1",
                "data":"cfg:language:en",
                "message":{"message_id":"7","chat":{"id":-10042,"type":"supergroup"}},
                "from":{"id":"99","language_code":"es"}
            })),
            Ok(CallbackContextOutcome::Context {
                context: CallbackContext {
                    callback_id: Some("callback-1".to_owned()),
                    data: "cfg:language:en".to_owned(),
                    chat_id: "-10042".to_owned(),
                    chat_type: "supergroup".to_owned(),
                    message_id: 7,
                    user_id: Some(99),
                    user_language_code: Some("es".to_owned()),
                    route: CallbackRoute::Config,
                }
            })
        );
        for (data, route) in [
            ("topup:p50", CallbackRoute::Topup),
            ("chg:1:2", CallbackRoute::Charges),
            ("task:delete:id", CallbackRoute::Task),
            ("sig:refresh:id", CallbackRoute::Signal),
            ("cfg:link:off", CallbackRoute::Config),
            ("other", CallbackRoute::Unknown),
        ] {
            assert_eq!(classify_callback(data), route);
        }
    }

    #[test]
    fn guard_outcome_covers_missing_required_callback_fields() {
        for value in [
            json!({}),
            json!({"data":""}),
            json!({"data":"cfg:x:y","message":{}}),
            json!({"data":"cfg:x:y","message":{"chat":{"id":null},"message_id":1}}),
            json!({"data":"cfg:x:y","message":{"chat":{"id":1}}}),
        ] {
            assert_eq!(
                parse_callback_context(&value),
                Ok(CallbackContextOutcome::Guard)
            );
        }
    }

    #[test]
    fn optional_fields_preserve_python_string_and_truthiness_semantics() {
        let actual = parse_callback_context(&json!({
            "id":0,
            "data":true,
            "message":{"message_id":2,"chat":{"id":true}},
            "from":{"id":"invalid","language_code":false}
        }));
        assert_eq!(
            actual,
            Ok(CallbackContextOutcome::Context {
                context: CallbackContext {
                    callback_id: None,
                    data: "True".to_owned(),
                    chat_id: "True".to_owned(),
                    chat_type: String::new(),
                    message_id: 2,
                    user_id: None,
                    user_language_code: None,
                    route: CallbackRoute::Unknown,
                }
            })
        );
    }

    #[test]
    fn malformed_nested_objects_and_message_ids_use_the_python_fallback() {
        assert_eq!(
            parse_callback_context(&json!([])),
            Err(CallbackParseError::InvalidCallback)
        );
        for value in [
            json!({"data":"cfg:x:y","message":"bad"}),
            json!({"data":"cfg:x:y","message":{"chat":"bad","message_id":1}}),
            json!({"data":"cfg:x:y","message":{"chat":{"id":1},"message_id":1},"from":"bad"}),
        ] {
            assert_eq!(
                parse_callback_context(&value),
                Err(CallbackParseError::InvalidNestedObject)
            );
        }
        assert_eq!(
            parse_callback_context(&json!({
                "data":"cfg:x:y",
                "message":{"chat":{"id":1},"message_id":"1.5"}
            })),
            Err(CallbackParseError::InvalidMessageId)
        );
    }
}
