//! Telegram command parsing without routing or I/O.

/// A normalized command token and its remaining message text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedCommand {
    /// Lowercase command token with the configured bot suffix removed.
    pub command: String,
    /// Remaining text after the first literal space, left-trimmed.
    pub message_text: String,
}

impl ParsedCommand {
    fn empty() -> Self {
        Self {
            command: String::new(),
            message_text: String::new(),
        }
    }
}

/// Match the existing Python command normalization behavior.
#[must_use]
pub fn parse_command(message_text: &str, bot_name: &str) -> ParsedCommand {
    let trimmed = message_text.trim();
    if trimmed.is_empty() {
        return ParsedCommand::empty();
    }

    let (command_token, remaining) = match trimmed.split_once(' ') {
        Some((command, remaining)) => (command, remaining.trim_start()),
        None => (trimmed, ""),
    };
    let mut command = command_token.to_lowercase().replace(bot_name, "");
    if let Some(command_body) = command.strip_prefix('/')
        && !command_body.is_empty()
        && command_body
            .chars()
            .all(|character| character == '\u{3164}')
    {
        command = "/ask".to_owned();
    }

    ParsedCommand {
        command,
        message_text: remaining.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use serde::Deserialize;

    use super::{ParsedCommand, parse_command};

    #[derive(Debug, Deserialize)]
    struct Contract {
        cases: Vec<Case>,
    }

    #[derive(Debug, Deserialize)]
    struct Case {
        input: String,
        bot_name: String,
        command: String,
        message_text: String,
    }

    fn contract() -> Result<Contract, serde_json::Error> {
        serde_json::from_str(include_str!("../../../contracts/command_parsing.json"))
    }

    #[test]
    fn matches_shared_contract() -> Result<(), serde_json::Error> {
        for case in contract()?.cases {
            assert_eq!(
                parse_command(&case.input, &case.bot_name),
                ParsedCommand {
                    command: case.command,
                    message_text: case.message_text,
                },
                "input={:?}",
                case.input
            );
        }
        Ok(())
    }

    proptest! {
        #[test]
        fn arbitrary_input_never_produces_leading_message_whitespace(
            input in ".{0,512}",
            bot_name in "@[a-z]{1,32}"
        ) {
            let parsed = parse_command(&input, &bot_name);
            prop_assert_eq!(parsed.message_text.trim_start(), parsed.message_text.as_str());
        }
    }
}
