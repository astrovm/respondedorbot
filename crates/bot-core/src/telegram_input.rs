//! Typed, side-effect-free parsing for incoming Telegram message payloads.

use serde::Serialize;
use serde_json::{Map, Value};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ChatId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct UserId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct MessageId(pub i64);

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MessageContent {
    pub text: String,
    pub photo_file_id: Option<String>,
    pub audio_file_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TelegramInputError {
    #[error("Telegram message payload must be an object")]
    InvalidMessage,
    #[error("Telegram media payload is malformed")]
    InvalidMedia,
    #[error("Telegram poll payload is malformed")]
    InvalidPoll,
}

fn python_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64() != Some(0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn poll_text(poll: &Map<String, Value>) -> Result<String, TelegramInputError> {
    let question = poll
        .get("question")
        .map_or_else(String::new, python_string)
        .trim()
        .to_owned();
    let mut options = Vec::new();
    match poll.get("options") {
        Some(Value::Array(raw_options)) => {
            for option in raw_options {
                let Some(option) = option.as_object() else {
                    continue;
                };
                if let Some(text) = option.get("text").filter(|value| python_truthy(value)) {
                    options.push(python_string(text).trim().to_owned());
                }
            }
        }
        Some(value) if python_truthy(value) => return Err(TelegramInputError::InvalidPoll),
        Some(_) | None => {}
    }
    if options.is_empty() {
        return Ok(question);
    }
    let options = options
        .into_iter()
        .map(|option| format!("- {option}"))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(if question.is_empty() {
        format!("Opciones:\n{options}")
    } else {
        format!("{question}\nOpciones:\n{options}")
    })
}

fn message_text(message: &Map<String, Value>) -> Result<String, TelegramInputError> {
    let mut parts = Vec::new();
    for field in ["text", "caption"] {
        if let Some(value) = message.get(field).filter(|value| python_truthy(value)) {
            parts.push(python_string(value).trim().to_owned());
        }
    }
    if let Some(Value::Object(poll)) = message.get("poll") {
        let text = poll_text(poll)?;
        if !text.is_empty() {
            parts.push(text);
        }
    }
    Ok(parts.join("\n\n"))
}

fn file_id(media: &Value) -> Result<String, TelegramInputError> {
    let media = media.as_object().ok_or(TelegramInputError::InvalidMedia)?;
    let value = media
        .get("file_id")
        .ok_or(TelegramInputError::InvalidMedia)?;
    Ok(python_string(value))
}

fn sticker_file_id(sticker: &Value) -> Result<Option<String>, TelegramInputError> {
    let sticker = sticker
        .as_object()
        .ok_or(TelegramInputError::InvalidMedia)?;
    let animated = sticker.get("is_animated").is_some_and(python_truthy)
        || sticker.get("is_video").is_some_and(python_truthy);
    if animated {
        let thumbnail = sticker.get("thumbnail").or_else(|| sticker.get("thumb"));
        if let Some(Value::Object(thumbnail)) = thumbnail
            && let Some(value) = thumbnail
                .get("file_id")
                .filter(|value| python_truthy(value))
        {
            return Ok(Some(python_string(value)));
        }
    }
    Ok(sticker
        .get("file_id")
        .filter(|value| python_truthy(value))
        .map(python_string))
}

fn visual_file_id(message: &Map<String, Value>) -> Result<Option<String>, TelegramInputError> {
    if let Some(photo) = message.get("photo").filter(|value| python_truthy(value)) {
        let photo = photo
            .as_array()
            .and_then(|items| items.last())
            .ok_or(TelegramInputError::InvalidMedia)?;
        return file_id(photo).map(Some);
    }
    if let Some(sticker) = message.get("sticker").filter(|value| python_truthy(value)) {
        return sticker_file_id(sticker);
    }
    let replied = match message.get("reply_to_message") {
        Some(Value::Object(replied)) => replied,
        Some(value) if python_truthy(value) => return Err(TelegramInputError::InvalidMedia),
        Some(_) | None => return Ok(None),
    };
    if let Some(photo) = replied.get("photo").filter(|value| python_truthy(value)) {
        let photo = photo
            .as_array()
            .and_then(|items| items.last())
            .ok_or(TelegramInputError::InvalidMedia)?;
        return file_id(photo).map(Some);
    }
    if let Some(sticker) = replied.get("sticker").filter(|value| python_truthy(value)) {
        return sticker_file_id(sticker);
    }
    Ok(None)
}

fn audio_file_id(message: &Map<String, Value>) -> Result<Option<String>, TelegramInputError> {
    const MEDIA_TYPES: [&str; 4] = ["voice", "audio", "video", "video_note"];
    for media_type in MEDIA_TYPES {
        if let Some(media) = message.get(media_type).filter(|value| python_truthy(value)) {
            return file_id(media).map(Some);
        }
    }
    let replied = match message.get("reply_to_message") {
        Some(Value::Object(replied)) => replied,
        Some(value) if python_truthy(value) => return Err(TelegramInputError::InvalidMedia),
        Some(_) | None => return Ok(None),
    };
    for media_type in MEDIA_TYPES {
        if let Some(media) = replied.get(media_type).filter(|value| python_truthy(value)) {
            return file_id(media).map(Some);
        }
    }
    Ok(None)
}

pub fn extract_message_content(message: &Value) -> Result<MessageContent, TelegramInputError> {
    let message = message
        .as_object()
        .ok_or(TelegramInputError::InvalidMessage)?;
    Ok(MessageContent {
        text: message_text(message)?,
        photo_file_id: visual_file_id(message)?,
        audio_file_id: audio_file_id(message)?,
    })
}

#[must_use]
pub fn is_group_chat_type(chat_type: Option<&str>) -> bool {
    matches!(chat_type, Some("group" | "supergroup"))
}

#[must_use]
pub fn normalize_numeric_id(value: &Value) -> Option<i64> {
    match value {
        Value::Bool(true) => Some(1),
        Value::Bool(false) => Some(0),
        Value::Number(value) => value
            .as_i64()
            .or_else(|| value.as_f64().map(|number| number.trunc() as i64)),
        Value::String(value) => value.trim().parse().ok(),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

#[must_use]
pub fn extract_user_id(message: &Value) -> Option<UserId> {
    message
        .as_object()?
        .get("from")?
        .as_object()?
        .get("id")
        .and_then(normalize_numeric_id)
        .map(UserId)
}

#[must_use]
pub fn format_user_identity(user: &Value) -> String {
    let Some(user) = user.as_object() else {
        return String::new();
    };
    let first_name = user
        .get("first_name")
        .filter(|value| !value.is_null())
        .map_or_else(String::new, python_string);
    let username = user
        .get("username")
        .filter(|value| !value.is_null())
        .map_or_else(String::new, python_string);
    if username.is_empty() {
        first_name
    } else {
        format!("{first_name} ({username})")
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        MessageContent, UserId, extract_message_content, extract_user_id, format_user_identity,
        is_group_chat_type, normalize_numeric_id,
    };

    #[test]
    fn extracts_text_caption_poll_and_direct_media_in_priority_order() {
        assert_eq!(
            extract_message_content(&json!({
                "text": "  hola  ",
                "caption": " mundo ",
                "poll": {"question": " Elegí ", "options": [{"text":" Uno "}, {"text":"Dos"}]},
                "photo": [{"file_id":"small"}, {"file_id":"large"}],
                "audio": {"file_id":"audio"},
                "video": {"file_id":"video"}
            })),
            Ok(MessageContent {
                text: "hola\n\nmundo\n\nElegí\nOpciones:\n- Uno\n- Dos".to_owned(),
                photo_file_id: Some("large".to_owned()),
                audio_file_id: Some("audio".to_owned()),
            })
        );
    }

    #[test]
    fn extracts_animated_sticker_thumbnail_and_replied_video() {
        assert_eq!(
            extract_message_content(&json!({
                "sticker": {"is_animated": true, "file_id":"animated", "thumbnail":{"file_id":"thumb"}},
                "reply_to_message": {"video":{"file_id":"replied-video"}}
            })),
            Ok(MessageContent {
                text: String::new(),
                photo_file_id: Some("thumb".to_owned()),
                audio_file_id: Some("replied-video".to_owned()),
            })
        );
    }

    #[test]
    fn rejects_malformed_media_and_poll_without_panicking() {
        assert!(extract_message_content(&json!([])).is_err());
        assert!(extract_message_content(&json!({"photo":[1]})).is_err());
        assert!(extract_message_content(&json!({"poll":{"options":{"bad":true}}})).is_err());
    }

    #[test]
    fn normalizes_ids_groups_and_python_style_identity() {
        assert_eq!(normalize_numeric_id(&json!(" -100123 ")), Some(-100123));
        assert_eq!(normalize_numeric_id(&json!(12.9)), Some(12));
        assert_eq!(
            extract_user_id(&json!({"from":{"id":"42"}})),
            Some(UserId(42))
        );
        assert_eq!(
            format_user_identity(&json!({"first_name":"Ana","username":"ana"})),
            "Ana (ana)"
        );
        assert_eq!(format_user_identity(&json!({"first_name":true})), "True");
        assert!(is_group_chat_type(Some("group")));
        assert!(is_group_chat_type(Some("supergroup")));
        assert!(!is_group_chat_type(Some("private")));
    }
}
