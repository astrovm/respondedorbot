//! Deterministic message-routing decisions.

/// Normalized facts needed to decide automatic media processing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaRoutingInput {
    pub chat_type: String,
    pub known_command: bool,
    pub message_text: String,
    pub bot_username: Option<String>,
    pub reply_username: Option<String>,
}

/// Decide whether media should be transcribed or described automatically.
#[must_use]
pub fn should_auto_process_media(input: &MediaRoutingInput) -> bool {
    if input.chat_type == "private" || input.known_command {
        return true;
    }

    let Some(bot_username) = input
        .bot_username
        .as_deref()
        .map(str::trim)
        .filter(|username| !username.is_empty())
    else {
        return false;
    };
    let bot_name = format!("@{bot_username}").to_lowercase();
    let is_mention = input.message_text.to_lowercase().contains(&bot_name);
    let is_reply = input.reply_username.as_deref() == Some(bot_username);
    is_mention || is_reply
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{MediaRoutingInput, should_auto_process_media};

    fn input() -> MediaRoutingInput {
        MediaRoutingInput {
            chat_type: "group".to_owned(),
            known_command: false,
            message_text: "ordinary message".to_owned(),
            bot_username: Some("testbot".to_owned()),
            reply_username: None,
        }
    }

    #[test]
    fn private_chats_and_known_commands_always_process_media() {
        let mut private = input();
        private.chat_type = "private".to_owned();
        private.bot_username = None;
        assert!(should_auto_process_media(&private));

        let mut command = input();
        command.known_command = true;
        command.bot_username = None;
        assert!(should_auto_process_media(&command));
    }

    #[test]
    fn groups_require_a_mention_or_exact_reply_username() {
        let mut mention = input();
        mention.message_text = "hola @TESTBOT".to_owned();
        assert!(should_auto_process_media(&mention));

        let mut reply = input();
        reply.reply_username = Some("testbot".to_owned());
        assert!(should_auto_process_media(&reply));

        let mut different_case_reply = input();
        different_case_reply.reply_username = Some("TestBot".to_owned());
        assert!(!should_auto_process_media(&different_case_reply));
    }

    #[test]
    fn missing_or_blank_username_disables_group_matching() {
        for username in [None, Some(String::new()), Some("  ".to_owned())] {
            let mut value = input();
            value.bot_username = username;
            value.message_text = "@testbot".to_owned();
            value.reply_username = Some("testbot".to_owned());
            assert!(!should_auto_process_media(&value));
        }
    }

    proptest! {
        #[test]
        fn arbitrary_text_and_usernames_never_panic(
            text in ".{0,512}",
            bot_username in ".{0,64}",
            reply_username in proptest::option::of(".{0,64}")
        ) {
            let value = MediaRoutingInput {
                chat_type: "group".to_owned(),
                known_command: false,
                message_text: text,
                bot_username: Some(bot_username),
                reply_username,
            };
            let _decision = should_auto_process_media(&value);
        }
    }
}
