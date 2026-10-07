//! Typed, side-effect-free parsing for incoming Telegram message payloads.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageId(pub i64);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

pub(crate) fn python_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64() != Some(0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

pub(crate) fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn vote_count(value: Option<&Value>) -> Option<u64> {
    value.and_then(Value::as_u64)
}

fn votes_label(count: u64) -> String {
    if count == 1 {
        "1 voto".to_owned()
    } else {
        format!("{count} votos")
    }
}

/// Renders a poll for the model. Telegram sends a fresh poll with zero votes,
/// so counts only show once someone voted or the poll closed; otherwise a
/// stored copy would claim nobody voted forever. Bots never see who voted in
/// polls they did not send, so the text says so instead of letting the model
/// guess.
fn poll_text(poll: &Map<String, Value>) -> Result<String, TelegramInputError> {
    let question = poll
        .get("question")
        .map_or_else(String::new, python_string)
        .trim()
        .to_owned();
    let closed = poll.get("is_closed").is_some_and(python_truthy);
    let total_votes = vote_count(poll.get("total_voter_count"));
    let show_counts = total_votes.is_some_and(|total| total > 0 || closed);
    let mut options = Vec::new();
    match poll.get("options") {
        Some(Value::Array(raw_options)) => {
            for option in raw_options {
                let Some(option) = option.as_object() else {
                    continue;
                };
                if let Some(text) = option.get("text").filter(|value| python_truthy(value)) {
                    let text = python_string(text).trim().to_owned();
                    options.push(
                        match vote_count(option.get("voter_count")).filter(|_| show_counts) {
                            Some(votes) => format!("- {text} ({})", votes_label(votes)),
                            None => format!("- {text}"),
                        },
                    );
                }
            }
        }
        Some(value) if python_truthy(value) => return Err(TelegramInputError::InvalidPoll),
        Some(_) | None => {}
    }
    if options.is_empty() {
        return Ok(question);
    }
    let mut lines = Vec::new();
    if !question.is_empty() {
        lines.push(question);
    }
    lines.push("Opciones:".to_owned());
    lines.extend(options);
    if let Some(total) = total_votes.filter(|_| show_counts) {
        lines.push(format!("Total: {}", votes_label(total)));
    }
    if closed {
        lines.push("Encuesta cerrada".to_owned());
    }
    match poll.get("is_anonymous") {
        Some(Value::Bool(true)) => lines.push("Encuesta anónima: nadie ve quién votó".to_owned()),
        Some(Value::Bool(false)) => {
            lines.push("Telegram no le muestra al bot quién votó".to_owned());
        }
        _ => {}
    }
    Ok(lines.join("\n"))
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
    if let Some(animation) = message
        .get("animation")
        .filter(|value| python_truthy(value))
    {
        return file_id(animation).map(Some);
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
    if let Some(animation) = replied
        .get("animation")
        .filter(|value| python_truthy(value))
    {
        return file_id(animation).map(Some);
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
        MessageContent, TelegramInputError, UserId, extract_message_content, extract_user_id,
        format_user_identity, is_group_chat_type, normalize_numeric_id,
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
    fn extracts_direct_and_replied_telegram_animations_as_visual_media() {
        for payload in [
            json!({"animation": {"file_id": "synthetic-animation"}}),
            json!({
                "reply_to_message": {
                    "animation": {"file_id": "synthetic-animation"}
                }
            }),
        ] {
            assert_eq!(
                extract_message_content(&payload),
                Ok(MessageContent {
                    text: String::new(),
                    photo_file_id: Some("synthetic-animation".to_owned()),
                    audio_file_id: None,
                })
            );
        }
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

    #[test]
    fn polls_follow_python_truthiness_for_questions_and_options() {
        assert_eq!(
            extract_message_content(&json!({
                "text": false,
                "caption": 0,
                "poll": {"question": null}
            })),
            Ok(MessageContent {
                text: "None".to_owned(),
                photo_file_id: None,
                audio_file_id: None,
            })
        );
        assert_eq!(
            extract_message_content(&json!({
                "poll": {"question": "", "options": ["raw", {"text": ""}, {"text": null}, {"text": false}, {"text": 5}]}
            }))
            .map(|content| content.text),
            Ok("Opciones:\n- 5".to_owned())
        );
        assert_eq!(
            extract_message_content(&json!({"poll": {"question": " ", "options": []}}))
                .map(|content| content.text),
            Ok(String::new())
        );
    }

    #[test]
    fn polls_show_votes_only_once_someone_voted_or_it_closed() {
        let poll = |total: u64, closed: bool, anonymous: bool| {
            extract_message_content(&json!({
                "poll": {
                    "question": "¿Qué hacés?",
                    "options": [
                        {"text": "Estoy laburando", "voter_count": total.saturating_sub(1)},
                        {"text": "Durmiendo", "voter_count": total.min(1)},
                        {"text": "Nada", "voter_count": 0}
                    ],
                    "total_voter_count": total,
                    "is_closed": closed,
                    "is_anonymous": anonymous
                }
            }))
            .map(|content| content.text)
        };
        assert_eq!(
            poll(0, false, false),
            Ok(
                "¿Qué hacés?\nOpciones:\n- Estoy laburando\n- Durmiendo\n- Nada\n\
                Telegram no le muestra al bot quién votó"
                    .to_owned()
            )
        );
        assert_eq!(
            poll(3, false, true),
            Ok(
                "¿Qué hacés?\nOpciones:\n- Estoy laburando (2 votos)\n- Durmiendo (1 voto)\n\
                - Nada (0 votos)\nTotal: 3 votos\nEncuesta anónima: nadie ve quién votó"
                    .to_owned()
            )
        );
        assert_eq!(
            poll(0, true, false),
            Ok(
                "¿Qué hacés?\nOpciones:\n- Estoy laburando (0 votos)\n- Durmiendo (0 votos)\n\
                - Nada (0 votos)\nTotal: 0 votos\nEncuesta cerrada\n\
                Telegram no le muestra al bot quién votó"
                    .to_owned()
            )
        );
        assert_eq!(
            extract_message_content(&json!({
                "poll": {
                    "options": [{"text": "Uno", "voter_count": "lots"}],
                    "total_voter_count": -2,
                    "is_anonymous": "yes"
                }
            }))
            .map(|content| content.text),
            Ok("Opciones:\n- Uno".to_owned())
        );
    }

    #[test]
    fn stickers_fall_back_from_empty_thumbnails_to_their_own_file() {
        for (sticker, expected) in [
            (
                json!({"is_video": 1, "thumb": {"file_id": ""}, "file_id": "video-sticker"}),
                Some("video-sticker"),
            ),
            (json!({"file_id": "plain-sticker"}), Some("plain-sticker")),
            (json!({"is_animated": true, "thumbnail": "none"}), None),
        ] {
            assert_eq!(
                extract_message_content(&json!({ "sticker": sticker }))
                    .map(|content| content.photo_file_id),
                Ok(expected.map(str::to_owned))
            );
        }
        assert_eq!(
            extract_message_content(&json!({"sticker": ["not", "an", "object"]})),
            Err(TelegramInputError::InvalidMedia)
        );
    }

    #[test]
    fn replied_media_is_used_only_when_the_message_has_none() {
        let content = |message| extract_message_content(&message);
        assert_eq!(
            content(json!({"reply_to_message": {"photo": [{"file_id": "a"}, {"file_id": "b"}]}}))
                .map(|content| content.photo_file_id),
            Ok(Some("b".to_owned()))
        );
        assert_eq!(
            content(json!({"reply_to_message": {"sticker": {"file_id": "replied-sticker"}}}))
                .map(|content| content.photo_file_id),
            Ok(Some("replied-sticker".to_owned()))
        );
        assert_eq!(
            content(json!({"text": "hola", "reply_to_message": {"text": "chau"}})),
            Ok(MessageContent {
                text: "hola".to_owned(),
                photo_file_id: None,
                audio_file_id: None,
            })
        );
        assert_eq!(
            content(json!({"reply_to_message": ""})).map(|content| content.photo_file_id),
            Ok(None)
        );
        assert_eq!(
            content(json!({"reply_to_message": {"photo": [1]}})),
            Err(TelegramInputError::InvalidMedia)
        );
        // A truthy non-object reply is malformed for both visual and audio lookup.
        assert_eq!(
            content(json!({"reply_to_message": "junk"})),
            Err(TelegramInputError::InvalidMedia)
        );
        assert_eq!(
            content(json!({"photo": [{"file_id": "p"}], "reply_to_message": "junk"})),
            Err(TelegramInputError::InvalidMedia)
        );
        assert_eq!(
            content(json!({"voice": {"no_file_id": true}})),
            Err(TelegramInputError::InvalidMedia)
        );
    }

    #[test]
    fn identities_and_ids_coerce_like_the_legacy_python_bot() {
        assert_eq!(normalize_numeric_id(&json!(true)), Some(1));
        assert_eq!(normalize_numeric_id(&json!(false)), Some(0));
        for value in [json!(null), json!([1]), json!({"id": 1}), json!("x")] {
            assert_eq!(normalize_numeric_id(&value), None, "{value}");
        }
        assert_eq!(extract_user_id(&json!({"from": "someone"})), None);
        assert_eq!(format_user_identity(&json!("Ana")), "");
        assert_eq!(format_user_identity(&json!({"first_name": false})), "False");
        assert_eq!(
            format_user_identity(&json!({"first_name": ["Ana"], "username": {"u": 1}})),
            "[\"Ana\"] ({\"u\":1})"
        );
        assert_eq!(
            format_user_identity(&json!({"first_name": null, "username": null})),
            ""
        );
    }
}
