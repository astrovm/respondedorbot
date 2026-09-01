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

    use super::{ParsedCommand, parse_command};

    #[test]
    fn parses_commands_and_message_text() {
        let cases = [
            ("", "@gordo", "", ""),
            ("   ", "@gordo", "", ""),
            ("/ASK hola", "@gordo", "/ask", "hola"),
            ("/ask@gordo   che", "@gordo", "/ask", "che"),
            ("/ask@gordo@gordo x", "@gordo", "/ask", "x"),
            ("/ㅤ hola", "@gordo", "/ask", "hola"),
            ("/ㅤㅤ   hola", "@gordo", "/ask", "hola"),
            ("/unknown", "@gordo", "/unknown", ""),
            ("hello world", "@gordo", "hello", "world"),
            ("/ask\tquestion", "@gordo", "/ask\tquestion", ""),
            ("/ask\nquestion", "@gordo", "/ask\nquestion", ""),
            ("/ASK@GORDO hi", "@gordo", "/ask", "hi"),
        ];

        for (input, bot_name, command, message_text) in cases {
            assert_eq!(
                parse_command(input, bot_name),
                ParsedCommand {
                    command: command.to_owned(),
                    message_text: message_text.to_owned(),
                },
                "input={:?}",
                input
            );
        }
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
