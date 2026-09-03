//! Compatibility plans for state written by native command replies.

use serde::Serialize;
use thiserror::Error;

use crate::message_state::{
    MessageWritePlan, bot_message_metadata_key, chat_members_key, prepare_chat_member_payload,
    prepare_message_write,
};
use crate::telegram_input::{ChatId, MessageId, UserId};

pub const CHAT_STATE_TTL_SECONDS: i64 = 30 * 24 * 60 * 60;
pub const BOT_MESSAGE_METADATA_TTL_SECONDS: i64 = 3 * 24 * 60 * 60;
pub const BOT_MESSAGE_METADATA_SCHEMA_VERSION: u8 = 1;
pub const CHAT_HISTORY_WRITE_LIMIT: usize = 400;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncomingCommandState<'a> {
    pub chat_id: ChatId,
    pub message_id: MessageId,
    pub user_id: UserId,
    pub first_name: Option<&'a str>,
    pub username: Option<&'a str>,
    pub text: &'a str,
    pub is_group: bool,
    pub timestamp: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutgoingCommandState<'a> {
    pub chat_id: ChatId,
    pub incoming_message_id: MessageId,
    pub sent_message_id: Option<MessageId>,
    pub text: &'a str,
    pub command: &'a str,
    pub timestamp: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMemberWritePlan {
    pub key: String,
    pub user_id: String,
    pub payload: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncomingCommandWritePlan {
    pub message: MessageWritePlan,
    pub member: Option<ChatMemberWritePlan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BotMetadataWritePlan {
    pub key: String,
    pub payload: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutgoingCommandWritePlan {
    pub message: MessageWritePlan,
    pub metadata: Option<BotMetadataWritePlan>,
}

#[derive(Debug, Error)]
pub enum CommandStateError {
    #[error("could not encode compatible command state: {0}")]
    Serialization(#[from] serde_json::Error),
}

#[derive(Serialize)]
struct CommandMetadata<'a> {
    schema_version: u8,
    r#type: &'static str,
    command: &'a str,
    uses_ai: bool,
}

fn user_identity(first_name: Option<&str>, username: Option<&str>) -> String {
    let first_name = first_name.unwrap_or_default();
    let username = username.unwrap_or_default();
    if username.is_empty() {
        first_name.to_owned()
    } else {
        format!("{first_name} ({username})")
    }
}

pub fn prepare_incoming_command_state(
    input: IncomingCommandState<'_>,
) -> Result<IncomingCommandWritePlan, CommandStateError> {
    let chat_id = input.chat_id.0.to_string();
    let message_id = input.message_id.0.to_string();
    let user_id = input.user_id.0.to_string();
    let username = input.username.unwrap_or_default();
    let text = format!(
        "{}: {}",
        user_identity(input.first_name, input.username),
        input.text
    );
    let message = prepare_message_write(
        &chat_id,
        &message_id,
        &text,
        input.timestamp,
        Some("user"),
        Some(&user_id),
        Some(username),
        None,
        input.text.contains('@') || input.text.starts_with('/'),
    )?;
    let member = if input.is_group {
        Some(ChatMemberWritePlan {
            key: chat_members_key(&chat_id),
            user_id,
            payload: prepare_chat_member_payload(
                input.first_name.unwrap_or_default(),
                username,
                input.timestamp,
            )?,
        })
    } else {
        None
    };
    Ok(IncomingCommandWritePlan { message, member })
}

pub fn prepare_outgoing_command_state(
    input: OutgoingCommandState<'_>,
) -> Result<OutgoingCommandWritePlan, CommandStateError> {
    let chat_id = input.chat_id.0.to_string();
    let stored_message_id = format!(
        "bot_{}",
        input.sent_message_id.unwrap_or(input.incoming_message_id).0
    );
    let message = prepare_message_write(
        &chat_id,
        &stored_message_id,
        input.text,
        input.timestamp,
        Some("assistant"),
        None,
        None,
        None,
        false,
    )?;
    let metadata = input.sent_message_id.map(|message_id| {
        let message_id = message_id.0.to_string();
        serde_json::to_string(&CommandMetadata {
            schema_version: BOT_MESSAGE_METADATA_SCHEMA_VERSION,
            r#type: "command",
            command: input.command,
            uses_ai: false,
        })
        .map(|payload| BotMetadataWritePlan {
            key: bot_message_metadata_key(&chat_id, &message_id),
            payload,
        })
    });
    let metadata = metadata.transpose()?;
    Ok(OutgoingCommandWritePlan { message, metadata })
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::{
        IncomingCommandState, OutgoingCommandState, prepare_incoming_command_state,
        prepare_outgoing_command_state,
    };
    use crate::telegram_input::{ChatId, MessageId, UserId};

    #[test]
    fn incoming_plan_matches_user_history_and_group_member_contracts() {
        let plan = prepare_incoming_command_state(IncomingCommandState {
            chat_id: ChatId(-42),
            message_id: MessageId(7),
            user_id: UserId(88),
            first_name: Some("Synthetic"),
            username: Some("tester"),
            text: "/time",
            is_group: true,
            timestamp: 1_672_531_200,
        });
        assert!(plan.is_ok());
        let Ok(plan) = plan else { return };
        assert_eq!(plan.message.text, "Synthetic (tester): /time");
        assert_eq!(plan.message.user_id, "88");
        assert_eq!(plan.message.mentions_bot, "1");
        let Some(member) = plan.member else { return };
        assert_eq!(member.key, "chat_members:-42");
        let payload = serde_json::from_str::<Value>(&member.payload);
        assert_eq!(
            payload
                .ok()
                .and_then(|value| value.get("last_seen").cloned()),
            Some(Value::from(1_672_531_200))
        );
    }

    #[test]
    fn outgoing_plan_uses_delivery_id_and_command_metadata() {
        let plan = prepare_outgoing_command_state(OutgoingCommandState {
            chat_id: ChatId(-42),
            incoming_message_id: MessageId(7),
            sent_message_id: Some(MessageId(99)),
            text: "1672531200",
            command: "/time",
            timestamp: 1_672_531_200,
        });
        assert!(plan.is_ok());
        let Ok(plan) = plan else { return };
        assert_eq!(plan.message.message_id, "bot_99");
        assert_eq!(plan.message.role, "assistant");
        let Some(metadata) = plan.metadata else {
            return;
        };
        assert_eq!(metadata.key, "bot_message_meta:-42:99");
        assert_eq!(
            serde_json::from_str::<Value>(&metadata.payload).ok(),
            Some(serde_json::json!({
                "schema_version":1,
                "type":"command",
                "command":"/time",
                "uses_ai":false
            }))
        );
    }

    #[test]
    fn private_and_unconfirmed_replies_skip_optional_state() {
        let incoming = prepare_incoming_command_state(IncomingCommandState {
            chat_id: ChatId(1),
            message_id: MessageId(2),
            user_id: UserId(3),
            first_name: None,
            username: None,
            text: "hello",
            is_group: false,
            timestamp: 4,
        });
        assert!(incoming.is_ok());
        assert!(incoming.ok().is_some_and(|plan| plan.member.is_none()));
        let outgoing = prepare_outgoing_command_state(OutgoingCommandState {
            chat_id: ChatId(1),
            incoming_message_id: MessageId(2),
            sent_message_id: None,
            text: "reply",
            command: "/instance",
            timestamp: 4,
        });
        assert!(outgoing.is_ok());
        let Ok(outgoing) = outgoing else { return };
        assert_eq!(outgoing.message.message_id, "bot_2");
        assert_eq!(outgoing.metadata, None);
    }
}
