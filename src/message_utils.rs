use crate::{ai, chat_config::ChatConfig, models};
use chrono::FixedOffset;
use regex::Regex;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct ChatHistoryEntry {
    pub role: String,
    pub text: String,
}

pub fn build_ai_messages(
    message: &models::Message,
    history: &[ChatHistoryEntry],
    message_text: &str,
    reply_context: Option<&str>,
) -> Vec<ai::ChatMessage> {
    let mut messages = Vec::new();
    for entry in history {
        messages.push(ai::ChatMessage {
            role: entry.role.clone(),
            content: entry.text.clone(),
        });
    }

    let chat_type = message.chat.kind.as_str();
    let chat_title = if chat_type == "private" {
        None
    } else {
        message.chat.title.as_deref()
    };
    let user_name = format_user_identity(message.from.as_ref());
    let now = chrono::Utc::now();
    let tz = FixedOffset::west_opt(3 * 3600).unwrap_or_else(|| FixedOffset::east_opt(0).unwrap());
    let local = now.with_timezone(&tz);

    let mut context_parts = Vec::new();
    context_parts.push("CONTEXTO:".to_string());
    if let Some(title) = chat_title {
        context_parts.push(format!("- Chat: {chat_type} ({title})"));
    } else {
        context_parts.push(format!("- Chat: {chat_type}"));
    }
    if user_name.is_empty() {
        context_parts.push("- Usuario: (desconocido)".to_string());
    } else {
        context_parts.push(format!("- Usuario: {user_name}"));
    }
    context_parts.push(format!("- Hora: {}", local.format("%H:%M")));

    if let Some(context) = reply_context {
        context_parts.push("".to_string());
        context_parts.push("MENSAJE AL QUE RESPONDE:".to_string());
        context_parts.push(truncate_text(context, 512));
    }

    context_parts.push("".to_string());
    context_parts.push("MENSAJE:".to_string());
    context_parts.push(truncate_text(message_text, 512));
    context_parts.push("".to_string());
    context_parts.push("INSTRUCCIONES:".to_string());
    context_parts.push("- Mantené el personaje del gordo".to_string());
    context_parts.push("- Usá lenguaje coloquial argentino".to_string());

    messages.push(ai::ChatMessage {
        role: "user".to_string(),
        content: context_parts.join("\n"),
    });

    let keep_from = messages.len().saturating_sub(8);
    messages.into_iter().skip(keep_from).collect()
}

pub fn post_process_ai_response(
    raw: &str,
    contexts: &[Option<String>],
    user_identity: Option<&str>,
) -> String {
    let sanitized = sanitize_tool_artifacts(raw);
    let no_gordo = remove_gordo_prefix(&sanitized);
    let context_stripped = strip_leading_context(&no_gordo, contexts);
    let prefix_stripped = strip_user_identity_prefix(&context_stripped, user_identity);
    clean_duplicate_response(&prefix_stripped)
}

pub fn sanitize_tool_artifacts(text: &str) -> String {
    if let Some(pos) = text.find("[TOOL]") {
        return text[..pos].trim().to_string();
    }
    text.trim().to_string()
}

pub fn remove_gordo_prefix(text: &str) -> String {
    if text.trim().is_empty() {
        return String::new();
    }
    let re = Regex::new(r"(?i)^\s*gordo\b\s*:\s*").unwrap();
    let mut cleaned = Vec::new();
    for line in text.lines() {
        cleaned.push(re.replace(line, "").to_string());
    }
    cleaned.join("\n").trim().to_string()
}

pub fn clean_duplicate_response(response: &str) -> String {
    if response.trim().is_empty() {
        return response.to_string();
    }
    let mut cleaned_lines: Vec<String> = Vec::new();
    for line in response.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() && cleaned_lines.last().map(|v| v.as_str()) != Some(trimmed) {
            cleaned_lines.push(trimmed.to_string());
        }
    }
    let cleaned = cleaned_lines.join("\n");
    let mut cleaned_sentences: Vec<String> = Vec::new();
    for sentence in cleaned.split(". ") {
        let trimmed = sentence.trim();
        if !trimmed.is_empty()
            && cleaned_sentences.last().map(|v| v.as_str()) != Some(trimmed)
        {
            cleaned_sentences.push(trimmed.to_string());
        }
    }
    cleaned_sentences.join(". ").replace("..", ".")
}

pub fn strip_leading_context(response: &str, contexts: &[Option<String>]) -> String {
    if response.trim().is_empty() {
        return response.to_string();
    }
    let normalized: Vec<String> = contexts
        .iter()
        .filter_map(|c| c.as_ref())
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .collect();
    if normalized.is_empty() {
        return response.to_string();
    }
    let mut trimmed = response.to_string();
    let mut passes = 0;
    let max_passes = normalized.len();
    loop {
        let mut changed = false;
        passes += 1;
        for context in &normalized {
            if trimmed.to_lowercase().starts_with(&context.to_lowercase()) {
                trimmed = trimmed[context.len()..]
                    .trim_start_matches([' ', '\t', ':', '-', '\n'])
                    .to_string();
                changed = true;
                break;
            }
        }
        if !changed || passes >= max_passes {
            break;
        }
    }
    trimmed
}

pub fn strip_user_identity_prefix(response: &str, user_identity: Option<&str>) -> String {
    let Some(identity) = user_identity else {
        return response.to_string();
    };
    let identity = identity.trim();
    if identity.is_empty() {
        return response.to_string();
    }
    let pattern = format!(r"(?i)^\s*{}\s*:\s*", regex::escape(identity));
    let re = Regex::new(&pattern).unwrap();
    re.replace(response, "").to_string().trim_start().to_string()
}

pub fn format_user_identity(user: Option<&models::User>) -> String {
    let Some(user) = user else {
        return String::new();
    };
    let mut name = user.first_name.trim().to_string();
    if let Some(last) = user.last_name.as_deref() {
        let last = last.trim();
        if !last.is_empty() {
            if !name.is_empty() {
                name.push(' ');
            }
            name.push_str(last);
        }
    }
    if let Some(username) = user.username.as_deref() {
        let username = username.trim();
        if !username.is_empty() {
            if name.is_empty() {
                return username.to_string();
            }
            return format!("{name} ({username})");
        }
    }
    name
}

pub fn format_shared_by(user: Option<&models::User>) -> Option<String> {
    let user = user?;
    if let Some(username) = user.username.as_deref() {
        let username = username.trim();
        if !username.is_empty() {
            return Some(format!("@{username}"));
        }
    }
    let mut parts = Vec::new();
    if !user.first_name.trim().is_empty() {
        parts.push(user.first_name.trim().to_string());
    }
    if let Some(last) = user.last_name.as_deref() {
        if !last.trim().is_empty() {
            parts.push(last.trim().to_string());
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

fn describe_replied_message(reply: &models::Message) -> Option<String> {
    if let Some(text) = reply.text.as_deref() {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    if reply.photo.as_ref().is_some() {
        return Some("una foto sin texto".to_string());
    }
    if reply.sticker.as_ref().is_some() {
        return Some("un sticker".to_string());
    }
    if reply.voice.as_ref().is_some() {
        return Some("un audio de voz".to_string());
    }
    if reply.audio.as_ref().is_some() {
        return Some("un archivo de audio".to_string());
    }
    if reply.video.as_ref().is_some() {
        return Some("un video".to_string());
    }
    if reply.document.as_ref().is_some() {
        return Some("un archivo adjunto".to_string());
    }
    None
}

pub fn build_reply_context_text(message: &models::Message) -> Option<String> {
    let reply = message.reply_to_message.as_ref()?;
    let description = describe_replied_message(reply)?;
    let reply_user = format_user_identity(reply.from.as_ref());
    if reply_user.is_empty() {
        Some(description)
    } else {
        Some(format!("{reply_user}: {description}"))
    }
}

pub fn format_user_message(
    message: &models::Message,
    message_text: &str,
    reply_context: Option<&str>,
) -> String {
    let formatted_user = format_user_identity(message.from.as_ref());
    if let Some(context) = reply_context {
        if formatted_user.is_empty() {
            return format!("(en respuesta a {context}): {message_text}");
        }
        return format!("{formatted_user} (en respuesta a {context}): {message_text}");
    }
    if formatted_user.is_empty() {
        message_text.to_string()
    } else {
        format!("{formatted_user}: {message_text}")
    }
}

pub fn truncate_text(text: &str, max_length: usize) -> String {
    if max_length == 0 {
        return String::new();
    }
    if text.chars().count() <= max_length {
        return text.to_string();
    }
    if max_length <= 3 {
        return ".".repeat(max_length);
    }
    let truncated: String = text.chars().take(max_length - 3).collect();
    format!("{truncated}...")
}

pub fn build_link_keyboard(urls: &[String]) -> Value {
    let keyboard: Vec<Vec<Value>> = urls
        .iter()
        .map(|url| vec![serde_json::json!({"text": "Open in app", "url": url})])
        .collect();
    serde_json::json!({ "inline_keyboard": keyboard })
}

pub fn contains_url(text: &str) -> bool {
    let re = Regex::new(r"https?://\S+").unwrap();
    re.is_match(text)
}

pub fn is_group_chat(message: &models::Message) -> bool {
    matches!(message.chat.kind.as_str(), "group" | "supergroup")
}

pub fn gen_random(name: &str) -> String {
    let rand_res = rand::random::<bool>();
    let rand_name = rand::random::<u8>() % 3;
    let mut msg = if rand_res { "si" } else { "no" }.to_string();
    match rand_name {
        1 => {
            msg = format!("{msg} boludo");
        }
        2 => {
            if !name.trim().is_empty() {
                msg = format!("{msg} {name}");
            }
        }
        _ => {}
    }
    msg
}

pub fn should_gordo_respond_core(
    bot_username: &str,
    message: &models::Message,
    text: &str,
    chat_config: &ChatConfig,
    reply_metadata: Option<&serde_json::Value>,
    trigger_words: &[String],
    random_value: f64,
) -> bool {
    let chat_type = message.chat.kind.as_str();
    let is_private = chat_type == "private";

    let bot_name = if bot_username.is_empty() {
        String::new()
    } else {
        format!("@{}", bot_username)
    };

    if let Some(reply) = message.reply_to_message.as_ref() {
        if let Some(reply_user) = reply.from.as_ref() {
            if reply_user.username.as_deref() == Some(bot_username) {
                if let Some(meta) = reply_metadata {
                    if meta.get("type").and_then(|v| v.as_str()) == Some("link_fix") {
                        return false;
                    }
                }
                if let Some(reply_text) = reply.text.as_ref() {
                    let replacement_domains = [
                        "fxtwitter.com",
                        "fixupx.com",
                        "fxbsky.app",
                        "kkinstagram.com",
                        "rxddit.com",
                        "vxtiktok.com",
                    ];
                    if replacement_domains.iter().any(|d| reply_text.contains(d)) {
                        return false;
                    }
                }
            }
        }
    }

    let message_lower = text.to_lowercase();
    let is_mention = !bot_name.is_empty() && message_lower.contains(&bot_name.to_lowercase());
    let is_reply = message
        .reply_to_message
        .as_ref()
        .and_then(|reply| reply.from.as_ref())
        .and_then(|u| u.username.as_deref())
        == Some(bot_username);

    if is_reply && !chat_config.ai_command_followups {
        if let Some(meta) = reply_metadata {
            let meta_type = meta.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let uses_ai = meta.get("uses_ai").and_then(|v| v.as_bool()).unwrap_or(false);
            if meta_type == "command" && !uses_ai {
                return false;
            }
        }
    }

    let mut is_trigger = false;
    if chat_config.ai_random_replies
        && trigger_words
            .iter()
            .any(|w| message_lower.contains(&w.to_lowercase()))
        && random_value < 0.1
    {
        is_trigger = true;
    }

    is_private || is_mention || is_reply || is_trigger
}
