//! Native known-chat-members AI tool.

use bot_adapters::redis_message_state::RedisMessageState;
use bot_core::chat_members::{decode_chat_members, render_chat_members};
use bot_core::locale::Locale;
use bot_core::message_state::chat_members_key;

use crate::chat_tool_loop::ToolExecutionResult;
use crate::tool_output;
use crate::tool_requests::{ExternalToolExecutor, ExternalToolRequest};

pub trait ChatMemberSource {
    fn members(&mut self, chat_id: &str) -> Result<Vec<(String, String)>, String>;
}

impl ChatMemberSource for RedisMessageState {
    fn members(&mut self, chat_id: &str) -> Result<Vec<(String, String)>, String> {
        self.get_chat_members(&chat_members_key(chat_id))
            .map(|members| {
                members
                    .into_iter()
                    .map(|member| (member.user_id, member.payload))
                    .collect()
            })
            .map_err(|error| error.to_string())
    }
}

pub struct ChatMembersTool<Source, Now> {
    source: Source,
    now: Now,
    chat_id: String,
    locale: Locale,
}

impl<Source, Now> ChatMembersTool<Source, Now> {
    #[must_use]
    pub fn new(source: Source, now: Now, chat_id: &str, locale: Locale) -> Self {
        Self {
            source,
            now,
            chat_id: chat_id.to_owned(),
            locale,
        }
    }
}

impl<Source, Now> ExternalToolExecutor for ChatMembersTool<Source, Now>
where
    Source: ChatMemberSource,
    Now: FnMut() -> i64,
{
    fn execute(
        &mut self,
        request: ExternalToolRequest,
        _tool_call_id: &str,
    ) -> ToolExecutionResult {
        if request != ExternalToolRequest::GetChatMembers {
            return ToolExecutionResult::output(tool_output::incompatible(
                self.locale,
                "get_chat_members",
            ));
        }
        if self.chat_id.is_empty() {
            return ToolExecutionResult::output(tool_output::unavailable(
                self.locale,
                "get_chat_members",
            ));
        }
        match self.source.members(&self.chat_id) {
            Ok(entries) => ToolExecutionResult::output(render_chat_members(
                &decode_chat_members(&entries),
                (self.now)(),
                self.locale,
            )),
            Err(error) => ToolExecutionResult::with_diagnostics(
                render_chat_members(&[], (self.now)(), self.locale),
                vec![format!("chat member lookup failed: {error}")],
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use bot_adapters::redis_connection::RedisEndpoint;
    use bot_core::message_state::{chat_members_key, prepare_chat_member_payload};

    use super::*;

    struct Source(Result<Vec<(String, String)>, String>);

    impl ChatMemberSource for Source {
        fn members(&mut self, chat_id: &str) -> Result<Vec<(String, String)>, String> {
            assert_eq!(chat_id, "-100");
            self.0.clone()
        }
    }

    #[test]
    fn renders_current_members_with_an_injected_clock() {
        let mut tool = ChatMembersTool::new(
            Source(Ok(vec![(
                "7".to_owned(),
                r#"{"schema_version":1,"first_name":"Ana","username":"ana","last_seen":9400}"#
                    .to_owned(),
            )])),
            || 10_000,
            "-100",
            Locale::Es,
        );
        assert_eq!(
            tool.execute(ExternalToolRequest::GetChatMembers, "call")
                .output,
            "Miembros conocidos:\n- Ana (@ana) — visto hace 10 min"
        );
    }

    #[test]
    fn missing_context_source_failure_and_wrong_request_are_safe() {
        let mut unavailable = ChatMembersTool::new(Source(Ok(Vec::new())), || 0, "", Locale::En);
        assert_eq!(
            unavailable
                .execute(ExternalToolRequest::GetChatMembers, "call")
                .output,
            "tool 'get_chat_members' is unavailable"
        );

        let mut failed = ChatMembersTool::new(
            Source(Err("synthetic Redis failure".to_owned())),
            || 0,
            "-100",
            Locale::En,
        );
        let result = failed.execute(ExternalToolRequest::GetChatMembers, "call");
        assert_eq!(result.output, "I do not know anyone in this chat yet");
        assert!(result.diagnostics[0].contains("synthetic Redis failure"));
        assert_eq!(
            failed.execute(ExternalToolRequest::TaskList, "call").output,
            "tool 'get_chat_members' received an incompatible request"
        );
    }

    #[test]
    fn redis_member_source_reads_the_persistent_member_shape() -> Result<(), String> {
        let Some(port) = std::env::var("TEST_REDIS_PORT")
            .ok()
            .and_then(|value| value.parse().ok())
        else {
            return Ok(());
        };
        let endpoint = RedisEndpoint {
            host: std::env::var("TEST_REDIS_HOST").unwrap_or_else(|_| "127.0.0.1".to_owned()),
            port,
            password: std::env::var("TEST_REDIS_PASSWORD")
                .ok()
                .filter(|value| !value.is_empty()),
        };
        let mut state = RedisMessageState::new(&endpoint).map_err(|error| error.to_string())?;
        let chat_id = format!("synthetic-members-{}", std::process::id());
        let payload = prepare_chat_member_payload("Synthetic", "synthetic_user", 1_700_000_000)
            .map_err(|error| error.to_string())?;
        state
            .save_chat_member(&chat_members_key(&chat_id), "42", &payload, 60)
            .map_err(|error| error.to_string())?;
        let members = ChatMemberSource::members(&mut state, &chat_id)?;
        assert_eq!(members, vec![("42".to_owned(), payload)]);
        Ok(())
    }
}
