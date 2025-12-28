use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Update {
    pub message: Option<Message>,
    pub callback_query: Option<CallbackQuery>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Message {
    pub message_id: i64,
    pub chat: Chat,
    pub from: Option<User>,
    pub text: Option<String>,
    pub reply_to_message: Option<Box<Message>>,
    pub photo: Option<Vec<PhotoSize>>,
    pub voice: Option<FileRef>,
    pub audio: Option<FileRef>,
    pub sticker: Option<FileRef>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CallbackQuery {
    pub id: String,
    pub from: User,
    pub message: Option<Message>,
    pub data: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Chat {
    pub id: i64,
    #[serde(rename = "type")]
    pub kind: String,
    pub title: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct User {
    pub id: i64,
    pub first_name: String,
    pub username: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PhotoSize {
    pub file_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FileRef {
    pub file_id: String,
}
