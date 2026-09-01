//! Typed chat configuration shared by command routing and settings flows.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

pub const DEFAULT_TIMEZONE_OFFSET: i64 = -3;
pub const DEFAULT_CREDITLESS_USER_HOURLY_LIMIT: i64 = 5;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatConfig {
    pub language: String,
    pub link_mode: String,
    pub ai_command_followups: bool,
    pub ignore_link_fix_followups: bool,
    pub timezone_offset: i64,
    pub ai_random_replies: bool,
    pub creditless_user_hourly_limit: i64,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            language: "auto".to_owned(),
            link_mode: "reply".to_owned(),
            ai_command_followups: true,
            ignore_link_fix_followups: true,
            timezone_offset: DEFAULT_TIMEZONE_OFFSET,
            ai_random_replies: true,
            creditless_user_hourly_limit: DEFAULT_CREDITLESS_USER_HOURLY_LIMIT,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ChatConfigError {
    #[error("chat configuration must be a JSON object")]
    InvalidObject,
    #[error("chat configuration field '{field}' is not a valid integer")]
    InvalidInteger { field: &'static str },
}

fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn coerce_bool(value: Option<&Value>, default: bool) -> bool {
    match value {
        Some(Value::Bool(value)) => *value,
        Some(Value::String(value)) => match value.trim().to_lowercase().as_str() {
            "true" | "1" | "yes" | "on" | "enabled" => true,
            "false" | "0" | "no" | "off" | "disabled" => false,
            _ => default,
        },
        Some(Value::Number(value)) => value.as_f64().is_some_and(|value| value != 0.0),
        Some(Value::Null | Value::Array(_) | Value::Object(_)) | None => default,
    }
}

fn coerce_integer(
    values: &Map<String, Value>,
    field: &'static str,
    legacy_field: Option<&str>,
    default: i64,
) -> Result<i64, ChatConfigError> {
    let value = values
        .get(field)
        .or_else(|| legacy_field.and_then(|legacy| values.get(legacy)));
    let Some(value) = value else {
        return Ok(default);
    };
    match value {
        Value::Bool(true) => Ok(1),
        Value::Bool(false) => Ok(0),
        Value::Number(value) => value
            .as_i64()
            .or_else(|| value.as_f64().map(|value| value.trunc() as i64))
            .ok_or(ChatConfigError::InvalidInteger { field }),
        Value::String(value) => value
            .trim()
            .parse()
            .map_err(|_| ChatConfigError::InvalidInteger { field }),
        Value::Null | Value::Array(_) | Value::Object(_) => {
            Err(ChatConfigError::InvalidInteger { field })
        }
    }
}

impl ChatConfig {
    pub fn from_json(value: &Value) -> Result<Self, ChatConfigError> {
        let values = value.as_object().ok_or(ChatConfigError::InvalidObject)?;
        let defaults = Self::default();
        Ok(Self {
            language: values
                .get("language")
                .map_or(defaults.language, python_string),
            link_mode: values
                .get("link_mode")
                .map_or(defaults.link_mode, python_string),
            ai_command_followups: coerce_bool(
                values.get("ai_command_followups"),
                defaults.ai_command_followups,
            ),
            ignore_link_fix_followups: coerce_bool(
                values.get("ignore_link_fix_followups"),
                defaults.ignore_link_fix_followups,
            ),
            timezone_offset: coerce_integer(
                values,
                "timezone_offset",
                None,
                defaults.timezone_offset,
            )?,
            ai_random_replies: coerce_bool(
                values.get("ai_random_replies"),
                defaults.ai_random_replies,
            ),
            creditless_user_hourly_limit: coerce_integer(
                values,
                "creditless_user_hourly_limit",
                Some("creditless_user_daily_limit"),
                defaults.creditless_user_hourly_limit,
            )?,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{ChatConfig, ChatConfigError};

    #[test]
    fn defaults_match_the_existing_python_service() {
        assert_eq!(ChatConfig::from_json(&json!({})), Ok(ChatConfig::default()));
    }

    #[test]
    fn normalizes_all_supported_legacy_shapes() {
        assert_eq!(
            ChatConfig::from_json(&json!({
                "language": "en",
                "link_mode": "replace",
                "ai_command_followups": "off",
                "ignore_link_fix_followups": 0,
                "timezone_offset": "4",
                "ai_random_replies": "enabled",
                "creditless_user_daily_limit": 9.8
            })),
            Ok(ChatConfig {
                language: "en".to_owned(),
                link_mode: "replace".to_owned(),
                ai_command_followups: false,
                ignore_link_fix_followups: false,
                timezone_offset: 4,
                ai_random_replies: true,
                creditless_user_hourly_limit: 9,
            })
        );
    }

    #[test]
    fn invalid_integer_fields_are_explicit() {
        assert_eq!(
            ChatConfig::from_json(&json!({"timezone_offset": "later"})),
            Err(ChatConfigError::InvalidInteger {
                field: "timezone_offset"
            })
        );
        assert_eq!(
            ChatConfig::from_json(&json!([])),
            Err(ChatConfigError::InvalidObject)
        );
    }
}
