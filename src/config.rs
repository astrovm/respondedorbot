use std::env;

#[derive(Debug, Clone)]
pub struct BotConfig {
    pub system_prompt: String,
    pub trigger_words: Vec<String>,
}

pub fn load_bot_config() -> Result<BotConfig, String> {
    let system_prompt = env::var("BOT_SYSTEM_PROMPT")
        .map_err(|_| "BOT_SYSTEM_PROMPT environment variable is required".to_string())?;
    let trigger_words_raw = env::var("BOT_TRIGGER_WORDS")
        .map_err(|_| "BOT_TRIGGER_WORDS environment variable is required".to_string())?;

    let trigger_words = trigger_words_raw
        .split(',')
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .collect();

    Ok(BotConfig {
        system_prompt,
        trigger_words,
    })
}
