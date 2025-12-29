use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::storage::Storage;
use crate::telegram;

static CALLBACKS_CHECKED: AtomicBool = AtomicBool::new(false);

const CHAT_CONFIG_KEY_PREFIX: &str = "chat_config:";
const LEGACY_LINK_MODE_PREFIX: &str = "link_mode:";
const CHAT_ADMIN_STATUS_TTL: u64 = 300;

#[derive(Debug, Clone)]
pub struct ChatConfig {
    pub link_mode: String,
    pub ai_random_replies: bool,
    pub ai_command_followups: bool,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            link_mode: "off".to_string(),
            ai_random_replies: true,
            ai_command_followups: true,
        }
    }
}

pub struct ConfigContext<'a> {
    pub http: &'a reqwest::Client,
    pub storage: &'a Storage,
    pub token: Option<&'a str>,
    pub webhook_key: Option<&'a str>,
    pub function_url: Option<&'a str>,
    pub admin_chat_id: Option<i64>,
}

pub async fn handle_config_command(
    ctx: &ConfigContext<'_>,
    chat_id: i64,
) -> Result<(String, Value), String> {
    ensure_callback_updates_enabled(ctx).await;
    let config = get_chat_config(ctx.storage, chat_id).await;
    Ok((build_config_text(&config), build_config_keyboard(&config)))
}

pub async fn handle_callback_query(
    ctx: &ConfigContext<'_>,
    callback: &crate::models::CallbackQuery,
) -> bool {
    let Some(callback_data) = callback.data.as_deref() else {
        answer_callback(ctx, &callback.id).await;
        return true;
    };

    let Some(message) = callback.message.as_ref() else {
        answer_callback(ctx, &callback.id).await;
        return true;
    };

    let chat_id = message.chat.id;
    let message_id = message.message_id;
    let chat_type = message.chat.kind.as_str();

    let is_config_callback = callback_data.starts_with("cfg:");
    if is_config_callback && is_group_chat_type(chat_type) {
        let is_admin = is_chat_admin(ctx, chat_id, callback.from.id).await;
        if !is_admin {
            let denial = "Solo los admins pueden cambiar la config del gordo acá.";
            if let Some(token) = ctx.token {
                let _ = telegram::send_message(
                    ctx.http,
                    token,
                    chat_id,
                    denial,
                    None,
                    Some(message_id),
                )
                .await;
            }
            report_unauthorized(ctx, chat_id, chat_type, callback.from.username.as_deref()).await;
            answer_callback(ctx, &callback.id).await;
            return true;
        }
    }

    let parts: Vec<&str> = callback_data.splitn(3, ':').collect();
    if parts.len() < 3 {
        answer_callback(ctx, &callback.id).await;
        return true;
    }
    let action = parts[1];
    let value = parts[2];

    let mut config = get_chat_config(ctx.storage, chat_id).await;
    if action == "link" && matches!(value, "reply" | "delete" | "off") {
        config.link_mode = value.to_string();
    } else if action == "random" {
        config.ai_random_replies = !config.ai_random_replies;
    } else if action == "followups" {
        config.ai_command_followups = !config.ai_command_followups;
    }

    config = set_chat_config(ctx.storage, chat_id, config).await;

    let text = build_config_text(&config);
    let keyboard = build_config_keyboard(&config);

    if let Some(token) = ctx.token {
        let edited = telegram::edit_message_text(
            ctx.http,
            token,
            chat_id,
            message_id,
            &text,
            &keyboard,
        )
        .await
        .unwrap_or(false);

        if !edited {
            let _ = telegram::send_message(ctx.http, token, chat_id, &text, Some(keyboard), None)
                .await;
        }
    }

    answer_callback(ctx, &callback.id).await;
    true
}

async fn answer_callback(ctx: &ConfigContext<'_>, callback_id: &str) {
    if let Some(token) = ctx.token {
        let _ = telegram::answer_callback_query(ctx.http, token, callback_id, None).await;
    }
}

fn chat_config_key(chat_id: i64) -> String {
    format!("{CHAT_CONFIG_KEY_PREFIX}{chat_id}")
}

fn legacy_link_mode_key(chat_id: i64) -> String {
    format!("{LEGACY_LINK_MODE_PREFIX}{chat_id}")
}

fn is_group_chat_type(chat_type: &str) -> bool {
    matches!(chat_type, "group" | "supergroup")
}

pub async fn get_chat_config(storage: &Storage, chat_id: i64) -> ChatConfig {
    let mut config = ChatConfig::default();

    if let Some(value) = storage.get_json(&chat_config_key(chat_id)).await {
        if let Some(obj) = value.as_object() {
            if let Some(mode) = obj.get("link_mode").and_then(|v| v.as_str()) {
                config.link_mode = mode.to_string();
            }
            config.ai_random_replies = coerce_bool(obj.get("ai_random_replies"), true);
            config.ai_command_followups = coerce_bool(obj.get("ai_command_followups"), true);
        }
        return config;
    }

    if let Some(legacy) = storage.get_string(&legacy_link_mode_key(chat_id)).await {
        if !legacy.trim().is_empty() {
            config.link_mode = legacy;
        }
    }

    config
}

pub async fn set_chat_config(
    storage: &Storage,
    chat_id: i64,
    config: ChatConfig,
) -> ChatConfig {
    let payload = json!({
        "link_mode": config.link_mode,
        "ai_random_replies": config.ai_random_replies,
        "ai_command_followups": config.ai_command_followups,
    });
    let _ = storage
        .set_json(&chat_config_key(chat_id), &payload, None)
        .await;

    let legacy_key = legacy_link_mode_key(chat_id);
    if matches!(config.link_mode.as_str(), "reply" | "delete") {
        let _ = storage
            .set_string(&legacy_key, &config.link_mode)
            .await;
    } else {
        let _ = storage.set_string(&legacy_key, "").await;
    }

    config
}

fn coerce_bool(value: Option<&Value>, default: bool) -> bool {
    match value {
        Some(Value::Bool(flag)) => *flag,
        Some(Value::String(text)) => {
            let lowered = text.trim().to_lowercase();
            match lowered.as_str() {
                "true" | "1" | "yes" | "on" | "enabled" => true,
                "false" | "0" | "no" | "off" | "disabled" => false,
                _ => default,
            }
        }
        Some(Value::Number(num)) => num.as_i64().unwrap_or(0) != 0,
        Some(Value::Null) | None => default,
        _ => default,
    }
}

pub fn build_config_text(config: &ChatConfig) -> String {
    let link_label = match config.link_mode.as_str() {
        "delete" => "Delete original message",
        "reply" => "Reply to original message",
        _ => "Off",
    };
    let random_label = if config.ai_random_replies { "✅ enabled" } else { "▫️ disabled" };
    let followups_label = if config.ai_command_followups { "✅ enabled" } else { "▫️ disabled" };

    format!(
        "Gordo config:\n\nLink fixer: {link_label}\nRandom AI replies: {random_label}\nFollow-ups for non-AI commands: {followups_label}\n\nUse the buttons below to change the settings.",
    )
}

pub fn build_config_keyboard(config: &ChatConfig) -> Value {
    let link_mode = config.link_mode.as_str();
    let random_enabled = config.ai_random_replies;
    let followups_enabled = config.ai_command_followups;

    json!({
        "inline_keyboard": [
            [
                build_choice_button("Reply to original message", "reply", link_mode, "link"),
                build_choice_button("Delete original message", "delete", link_mode, "link"),
                build_choice_button("Off", "off", link_mode, "link"),
            ],
            [build_toggle_button("Random AI replies", random_enabled, "random")],
            [build_toggle_button(
                "Follow-ups for non-AI commands",
                followups_enabled,
                "followups",
            )],
        ]
    })
}

fn build_choice_button(label: &str, value: &str, current: &str, action: &str) -> Value {
    let prefix = if current == value { "✅" } else { "▫️" };
    json!({
        "text": format!("{prefix} {label}"),
        "callback_data": format!("cfg:{action}:{value}"),
    })
}

fn build_toggle_button(label: &str, enabled: bool, action: &str) -> Value {
    let prefix = if enabled { "✅" } else { "▫️" };
    json!({
        "text": format!("{prefix} {label}"),
        "callback_data": format!("cfg:{action}:toggle"),
    })
}

pub async fn is_chat_admin(ctx: &ConfigContext<'_>, chat_id: i64, user_id: i64) -> bool {
    let Some(token) = ctx.token else {
        return false;
    };

    let cache_key = format!("chat_admin:{chat_id}:{user_id}");
    if let Some(cached) = ctx.storage.get_string(&cache_key).await {
        return cached == "1";
    }

    let status = telegram::get_chat_member_status(ctx.http, token, chat_id, user_id)
        .await
        .unwrap_or(None);
    let is_admin = matches!(status.as_deref(), Some("administrator") | Some("creator"));

    let cached_value = if is_admin { "1" } else { "0" };
    let _ = ctx
        .storage
        .set_string_with_ttl(&cache_key, CHAT_ADMIN_STATUS_TTL, cached_value)
        .await;

    is_admin
}

pub async fn report_unauthorized(
    ctx: &ConfigContext<'_>,
    chat_id: i64,
    chat_type: &str,
    username: Option<&str>,
) {
    tracing::warn!(
        chat_id,
        chat_type,
        username = username.unwrap_or(""),
        "Unauthorized config attempt"
    );

    if let (Some(token), Some(admin_chat_id)) = (ctx.token, ctx.admin_chat_id) {
        let msg = format!(
            "Unauthorized config attempt: chat_id={chat_id} chat_type={chat_type} username={}",
            username.unwrap_or(""),
        );
        let _ = telegram::send_message(ctx.http, token, admin_chat_id, &msg, None, None).await;
    }
}

async fn ensure_callback_updates_enabled(ctx: &ConfigContext<'_>) {
    if CALLBACKS_CHECKED.swap(true, Ordering::SeqCst) {
        return;
    }

    let (Some(token), Some(webhook_key), Some(function_url)) =
        (ctx.token, ctx.webhook_key, ctx.function_url)
    else {
        return;
    };

    let info = match telegram::get_webhook_info(ctx.http, token).await {
        Ok(info) => info,
        Err(_) => return,
    };

    let result = info.get("result").cloned().unwrap_or(Value::Null);
    let allowed = result.get("allowed_updates");
    let expected_url = format!("{function_url}?key={webhook_key}");
    let current_url = result.get("url").and_then(|value| value.as_str()).unwrap_or("");

    if let Some(Value::Array(list)) = allowed {
        if list.is_empty() || list.iter().any(|v| v.as_str() == Some("callback_query")) {
            return;
        }
    } else if allowed.is_none() {
        return;
    }

    if !current_url.is_empty() && current_url != expected_url {
        return;
    }

    let _ = telegram::set_webhook(ctx.http, token, function_url, webhook_key, ctx.storage).await;
}

pub async fn increment_rate_limit(
    storage: &Storage,
    key: &str,
    ttl_seconds: u64,
) -> i64 {
    storage.incr_with_ttl(key, ttl_seconds).await
}
