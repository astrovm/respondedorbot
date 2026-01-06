use rand::TryRngCore;
use sha2::{Digest, Sha256};

use crate::http::{HttpClient, HttpResult};
use crate::storage::Storage;

pub const TELEGRAM_SECRET_TOKEN_KEY: &str = "X-Telegram-Bot-Api-Secret-Token";
const TELEGRAM_SECRET_TOKEN_TTL: u64 = 90 * 24 * 60 * 60;

fn api_base(token: &str) -> String {
    format!("https://api.telegram.org/bot{token}")
}

pub async fn get_webhook_info(http: &HttpClient, token: &str) -> HttpResult<serde_json::Value> {
    let url = format!("{}/getWebhookInfo", api_base(token));
    let response = http.get(url).send().await?;
    response.json::<serde_json::Value>().await
}

pub async fn set_webhook(
    http: &HttpClient,
    token: &str,
    function_url: &str,
    webhook_key: &str,
    storage: &Storage,
) -> HttpResult<bool> {
    let url = format!("{}/setWebhook", api_base(token));

    let secret_token = generate_secret_token();
    let full_url = format!("{}?key={}", function_url, webhook_key);

    let payload = serde_json::json!({
        "url": full_url,
        "allowed_updates": ["message", "callback_query"],
        "secret_token": secret_token,
        "max_connections": 8,
    });

    let response = http.post(url).json(&payload).send().await?;
    let body = response.json::<serde_json::Value>().await?;

    let ok = body
        .get("ok")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);

    if ok {
        let stored = storage
            .set_string_with_ttl(
                TELEGRAM_SECRET_TOKEN_KEY,
                TELEGRAM_SECRET_TOKEN_TTL,
                &secret_token,
            )
            .await
            || storage
                .set_string(TELEGRAM_SECRET_TOKEN_KEY, &secret_token)
                .await;

        if !stored {
            tracing::warn!("failed to persist Telegram webhook secret token");
        }
    }

    Ok(ok)
}

fn generate_secret_token() -> String {
    let mut bytes = [0u8; 32];
    let mut rng = rand::rngs::OsRng;
    rng.try_fill_bytes(&mut bytes)
        .expect("failed to generate secret token");
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    hex::encode(digest)
}

pub async fn send_message(
    http: &HttpClient,
    token: &str,
    chat_id: i64,
    text: &str,
    reply_markup: Option<serde_json::Value>,
    reply_to: Option<i64>,
) -> HttpResult<Option<i64>> {
    let url = format!("{}/sendMessage", api_base(token));
    let mut payload = serde_json::json!({
        "chat_id": chat_id,
        "text": text,
    });
    if text.to_lowercase().contains("polymarket.com") {
        payload["disable_web_page_preview"] = serde_json::Value::from(true);
    }
    if let Some(markup) = reply_markup {
        payload["reply_markup"] = markup;
    }
    if let Some(reply_to_id) = reply_to {
        payload["reply_to_message_id"] = serde_json::Value::from(reply_to_id);
    }
    let response = http.post(url).json(&payload).send().await?;
    let body = response.json::<serde_json::Value>().await?;

    let message_id = body
        .get("result")
        .and_then(|result| result.get("message_id"))
        .and_then(|value| value.as_i64());

    Ok(message_id)
}

pub async fn answer_callback_query(
    http: &HttpClient,
    token: &str,
    callback_id: &str,
    text: Option<&str>,
) -> HttpResult<bool> {
    let url = format!("{}/answerCallbackQuery", api_base(token));
    let payload = serde_json::json!({
        "callback_query_id": callback_id,
        "text": text.unwrap_or(""),
        "show_alert": false,
    });
    let response = http.post(url).json(&payload).send().await?;
    let body = response.json::<serde_json::Value>().await?;
    Ok(body
        .get("ok")
        .and_then(|value| value.as_bool())
        .unwrap_or(false))
}

pub async fn edit_message_text(
    http: &HttpClient,
    token: &str,
    chat_id: i64,
    message_id: i64,
    text: &str,
    reply_markup: &serde_json::Value,
) -> HttpResult<bool> {
    let url = format!("{}/editMessageText", api_base(token));
    let payload = serde_json::json!({
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "reply_markup": reply_markup,
    });
    let response = http.post(url).json(&payload).send().await?;
    let body = response.json::<serde_json::Value>().await?;
    Ok(body
        .get("ok")
        .and_then(|value| value.as_bool())
        .unwrap_or(false))
}

pub async fn send_chat_action(
    http: &HttpClient,
    token: &str,
    chat_id: i64,
    action: &str,
) -> HttpResult<bool> {
    let url = format!("{}/sendChatAction", api_base(token));
    let payload = serde_json::json!({
        "chat_id": chat_id,
        "action": action,
    });
    let response = http.post(url).json(&payload).send().await?;
    let body = response.json::<serde_json::Value>().await?;
    Ok(body.get("ok").and_then(|v| v.as_bool()).unwrap_or(false))
}

pub async fn get_chat_member_status(
    http: &HttpClient,
    token: &str,
    chat_id: i64,
    user_id: i64,
) -> HttpResult<Option<String>> {
    let url = format!("{}/getChatMember", api_base(token));
    let response = http
        .get(url)
        .query("chat_id", chat_id.to_string())
        .query("user_id", user_id.to_string())
        .send()
        .await?;
    let body = response.json::<serde_json::Value>().await?;
    if body.get("ok").and_then(|value| value.as_bool()) != Some(true) {
        return Ok(None);
    }
    let status = body
        .get("result")
        .and_then(|result| result.get("status"))
        .and_then(|value| value.as_str())
        .map(|value| value.to_string());
    Ok(status)
}

pub async fn delete_message(
    http: &HttpClient,
    token: &str,
    chat_id: i64,
    message_id: i64,
) -> HttpResult<bool> {
    let url = format!("{}/deleteMessage", api_base(token));
    let payload = serde_json::json!({
        "chat_id": chat_id,
        "message_id": message_id,
    });
    let response = http.post(url).json(&payload).send().await?;
    let body = response.json::<serde_json::Value>().await?;
    Ok(body.get("ok").and_then(|v| v.as_bool()).unwrap_or(false))
}

pub async fn get_file_path(
    http: &HttpClient,
    token: &str,
    file_id: &str,
) -> HttpResult<Option<String>> {
    let url = format!("{}/getFile", api_base(token));
    let response = http
        .get(url)
        .query("file_id", file_id)
        .send()
        .await?;
    let body = response.json::<serde_json::Value>().await?;
    if body.get("ok").and_then(|v| v.as_bool()) != Some(true) {
        return Ok(None);
    }
    let path = body
        .get("result")
        .and_then(|v| v.get("file_path"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    Ok(path)
}

pub async fn download_file(
    http: &HttpClient,
    token: &str,
    file_id: &str,
) -> Result<Vec<u8>, String> {
    let path = get_file_path(http, token, file_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "missing file path".to_string())?;
    let url = format!("https://api.telegram.org/file/bot{token}/{path}");
    let bytes = http
        .get(url)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .bytes()
        .await
        .map_err(|e| e.to_string())?;
    Ok(bytes.to_vec())
}
