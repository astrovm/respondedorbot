pub fn parse_command(message_text: &str, bot_username: Option<&str>) -> Option<(String, String)> {
    let trimmed = message_text.trim();
    if !trimmed.starts_with('/') {
        return None;
    }

    let mut parts = trimmed.splitn(2, ' ');
    let raw_command = parts.next().unwrap_or("");
    let args = parts.next().unwrap_or("").trim().to_string();

    let mut command = raw_command.to_string();
    if let Some((name, target)) = raw_command.split_once('@') {
        if let Some(expected) = bot_username {
            if !target.eq_ignore_ascii_case(expected) {
                return None;
            }
        }
        command = name.to_string();
    }

    Some((command, args))
}

pub fn help_text() -> String {
    "Comandos: /help, /start, /config, /prices, /dolar, /usd, /bcra, /search, /ask, /devo, /rulo, /powerlaw, /rainbow, /convertbase, /transcribe, /time".to_string()
}
