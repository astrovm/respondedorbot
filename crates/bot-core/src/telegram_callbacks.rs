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
    Market,
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
    pub ephemeral_message_id: Option<i64>,
    pub receiver_user_id: Option<i64>,
    pub user_id: Option<i64>,
    pub user_language_code: Option<String>,
    pub route: CallbackRoute,
}

impl CallbackContext {
    pub fn personal_menu_allowed(&self) -> bool {
        match self.user_id {
            Some(user) if self.chat_type == "private" => self.chat_id == user.to_string(),
            Some(user) if matches!(self.chat_type.as_str(), "group" | "supergroup") => {
                self.ephemeral_message_id.is_some() && self.receiver_user_id == Some(user)
            }
            Some(_) => false,
            None => false,
        }
    }

    pub fn personal_action(
        &self,
        action: crate::telegram_actions::TelegramAction,
    ) -> crate::telegram_actions::TelegramAction {
        if self.chat_type == "private" {
            action
        } else {
            crate::telegram_actions::TelegramAction::Ephemeral {
                receiver_user_id: self.user_id.unwrap_or_default(),
                ephemeral_message_id: self.ephemeral_message_id,
                action: Box::new(action),
            }
        }
    }
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
    } else if data.starts_with("mkt:") {
        CallbackRoute::Market
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
    let Some(message_id_value) = message
        .get("ephemeral_message_id")
        .or_else(|| message.get("message_id"))
        .filter(|value| !value.is_null())
    else {
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
            ephemeral_message_id: message.get("ephemeral_message_id").and_then(Value::as_i64),
            receiver_user_id: message
                .get("receiver_user")
                .and_then(|u| u.get("id"))
                .and_then(Value::as_i64),
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
                    ephemeral_message_id: None,
                    receiver_user_id: None,
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
            ("mkt:select:id:0", CallbackRoute::Market),
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
                    ephemeral_message_id: None,
                    receiver_user_id: None,
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

    #[test]
    fn falsy_nested_values_count_as_empty_objects() {
        assert_eq!(
            parse_callback_context(&json!({"data":"cfg:x:y","message":0})),
            parse_callback_context(&json!({"data":"cfg:x:y"}))
        );
    }

    #[test]
    fn personal_menu_authorization_requires_the_actual_receiver()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::telegram_actions::{SendMessage, TelegramAction};
        use crate::telegram_input::ChatId;
        for chat_type in ["private", "group", "supergroup", "channel", ""] {
            for user in [None, Some(42), Some(99)] {
                for receiver in [None, Some(42), Some(99)] {
                    for ephemeral in [None, Some(71)] {
                        let context = CallbackContext {
                            callback_id: None,
                            data: "topup:amt:50".into(),
                            chat_id: if chat_type == "private" { "42" } else { "-7" }.into(),
                            chat_type: chat_type.into(),
                            message_id: 7,
                            user_id: user,
                            receiver_user_id: receiver,
                            ephemeral_message_id: ephemeral,
                            user_language_code: None,
                            route: CallbackRoute::Topup,
                        };
                        let expected = match chat_type {
                            "private" => user == Some(42),
                            "group" | "supergroup" => {
                                user.is_some() && receiver == user && ephemeral.is_some()
                            }
                            _ => false,
                        };
                        assert_eq!(
                            context.personal_menu_allowed(),
                            expected,
                            "{chat_type} sender={user:?} receiver={receiver:?} ephemeral={ephemeral:?}"
                        );
                        let action = TelegramAction::SendMessage(SendMessage::new(
                            ChatId(-7),
                            "personal menu",
                        ));
                        if chat_type == "private" {
                            assert_eq!(context.personal_action(action.clone()), action);
                        } else {
                            assert_eq!(
                                context.personal_action(action.clone()),
                                TelegramAction::Ephemeral {
                                    receiver_user_id: user.unwrap_or_default(),
                                    ephemeral_message_id: ephemeral,
                                    action: Box::new(action)
                                }
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
