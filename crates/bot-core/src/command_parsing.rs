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

/// Parse a command and remove only an exact suffix addressed to the configured bot.
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
    let mut command = command_token.to_lowercase();
    let bot_username = bot_name.trim().trim_start_matches('@').to_lowercase();
    let bot_suffix = format!("@{bot_username}");
    if command.starts_with('/')
        && !bot_username.is_empty()
        && let Some(without_suffix) = command.strip_suffix(&bot_suffix)
    {
        command = without_suffix.to_owned();
    }
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
    use crate::locale::Locale;
    use crate::telegram_commands::telegram_commands;

    #[test]
    fn parses_commands_and_message_text() {
        let cases = [
            ("", "@gordo", "", ""),
            ("   ", "@gordo", "", ""),
            ("/ASK hola", "@gordo", "/ask", "hola"),
            ("/ask@gordo   che", "@gordo", "/ask", "che"),
            ("/ask@gordo   che", "gordo", "/ask", "che"),
            ("/ㅤ hola", "@gordo", "/ask", "hola"),
            ("/ㅤㅤ   hola", "@gordo", "/ask", "hola"),
            ("/unknown", "@gordo", "/unknown", ""),
            ("hello world", "@gordo", "hello", "world"),
            ("/ask\tquestion", "@gordo", "/ask\tquestion", ""),
            ("/ask\nquestion", "@gordo", "/ask\nquestion", ""),
            ("/ASK@GORDO hi", "@gordo", "/ask", "hi"),
            ("/ask@gordo@gordo x", "@gordo", "/ask@gordo", "x"),
            ("/gordo", "gordo", "/gordo", ""),
            ("hello@gordo world", "gordo", "hello@gordo", "world"),
            (
                "/balance@playtimbabot",
                "gordo",
                "/balance@playtimbabot",
                "",
            ),
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

    #[test]
    fn parses_every_public_and_admin_command_addressed_to_this_bot() {
        let mut commands = telegram_commands(Locale::Es)
            .into_iter()
            .map(|entry| entry.command)
            .collect::<Vec<_>>();
        commands.extend(["printcredits", "creditlog"]);

        for bot_name in ["respondedorbot", "@respondedorbot"] {
            for command in &commands {
                assert_eq!(
                    parse_command(&format!("/{command}@RespondedorBot value"), bot_name),
                    ParsedCommand {
                        command: format!("/{command}"),
                        message_text: "value".to_owned(),
                    },
                    "command={command} bot_name={bot_name}",
                );
            }
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
