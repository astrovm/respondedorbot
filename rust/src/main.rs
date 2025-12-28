use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    routing::get,
    Json, Router,
};
use serde::Deserialize;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod config;
mod redis_store;
mod telegram;
mod commands;
mod chat_config;
mod http_cache;
mod market;
mod ai;
mod agent;
mod bcra;
mod tools;
mod weather;
mod hacker_news;
mod media;
mod models;

use crate::models::Update;
use crate::redis_store::{create_redis_client, redis_get_string};

#[derive(Clone)]
struct AppState {
    http: reqwest::Client,
    redis: redis::Client,
    telegram_token: Option<String>,
    webhook_key: Option<String>,
    function_url: Option<String>,
    admin_chat_id: Option<i64>,
    bot_username: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WebhookQuery {
    key: Option<String>,
    check_webhook: Option<String>,
    update_webhook: Option<String>,
    update_dollars: Option<String>,
    run_agent: Option<String>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();

    let http = reqwest::Client::new();
    let redis_host = std::env::var("REDIS_HOST").unwrap_or_else(|_| "localhost".to_string());
    let redis_port: u16 = std::env::var("REDIS_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(6379);
    let redis_password = std::env::var("REDIS_PASSWORD").ok();
    let redis = create_redis_client(&redis_host, redis_port, redis_password.as_deref())
        .expect("failed to create redis client");

    let state = AppState {
        http,
        redis,
        telegram_token: std::env::var("TELEGRAM_TOKEN").ok(),
        webhook_key: std::env::var("WEBHOOK_AUTH_KEY").ok(),
        function_url: std::env::var("FUNCTION_URL").ok(),
        admin_chat_id: std::env::var("ADMIN_CHAT_ID").ok().and_then(|value| value.parse().ok()),
        bot_username: std::env::var("TELEGRAM_USERNAME").ok(),
    };

    match config::load_bot_config() {
        Ok(cfg) => {
            let prompt_len = cfg.system_prompt.len();
            tracing::info!(
                prompt_len,
                trigger_words = ?cfg.trigger_words,
                "bot config loaded"
            );
        }
        Err(err) => {
            tracing::warn!(error = %err, "bot config not loaded");
        }
    }

    let app = Router::new()
        .route("/", get(handle_get).post(handle_post))
        .with_state(state);

    let addr = "0.0.0.0:8080";
    tracing::info!("listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind address");
    axum::serve(listener, app).await.expect("server failed");
}

async fn handle_get(
    State(state): State<AppState>,
    Query(query): Query<WebhookQuery>,
) -> (StatusCode, String) {
    let Some(key) = query.key else {
        return (StatusCode::OK, "No key".to_string());
    };
    if state.webhook_key.as_deref() != Some(key.as_str()) {
        return (StatusCode::BAD_REQUEST, "Wrong key".to_string());
    }

    if is_true(&query.check_webhook) {
        return handle_check_webhook(&state).await;
    }

    if is_true(&query.update_webhook) {
        return handle_update_webhook(&state).await;
    }

    if is_true(&query.update_dollars) {
        return (StatusCode::NOT_IMPLEMENTED, "Not implemented".to_string());
    }

    if is_true(&query.run_agent) {
        return (StatusCode::NOT_IMPLEMENTED, "Not implemented".to_string());
    }

    (StatusCode::OK, "Ok".to_string())
}

async fn handle_post(
    State(state): State<AppState>,
    Query(query): Query<WebhookQuery>,
    headers: HeaderMap,
    body: Result<Json<Update>, axum::extract::rejection::JsonRejection>,
) -> (StatusCode, String) {
    let Some(key) = query.key else {
        return (StatusCode::OK, "No key".to_string());
    };
    if state.webhook_key.as_deref() != Some(key.as_str()) {
        return (StatusCode::BAD_REQUEST, "Wrong key".to_string());
    }

    if !is_secret_token_valid(&state, &headers).await {
        return (StatusCode::BAD_REQUEST, "Wrong secret token".to_string());
    }

    let update = match body {
        Ok(Json(value)) => value,
        Err(_) => return (StatusCode::BAD_REQUEST, "Invalid JSON".to_string()),
    };

    if let Some(callback) = update.callback_query {
        tracing::info!(callback_id = %callback.id, "callback query received");
        let ctx = config_context(&state);
        let _ = chat_config::handle_callback_query(&ctx, &callback).await;
        return (StatusCode::OK, "Ok".to_string());
    }

    let Some(message) = update.message else {
        return (StatusCode::OK, "No message".to_string());
    };

    if let Err(err) = handle_message(&state, message).await {
        tracing::error!(error = %err, "handle_message failed");
        admin_report(&state, "Error processing message").await;
    }
    (StatusCode::OK, "Ok".to_string())
}

async fn handle_check_webhook(state: &AppState) -> (StatusCode, String) {
    let Some(token) = state.telegram_token.as_deref() else {
        return (StatusCode::BAD_REQUEST, "Webhook check error".to_string());
    };
    let Some(function_url) = state.function_url.as_deref() else {
        return (StatusCode::BAD_REQUEST, "Webhook check error".to_string());
    };
    let Some(webhook_key) = state.webhook_key.as_deref() else {
        return (StatusCode::BAD_REQUEST, "Webhook check error".to_string());
    };

    match telegram::get_webhook_info(&state.http, token).await {
        Ok(info) => {
            let expected = format!("{function_url}?key={webhook_key}");
            let current = info
                .get("result")
                .and_then(|result| result.get("url"))
                .and_then(|url| url.as_str())
                .unwrap_or("");

            if current == expected {
                (StatusCode::OK, "Webhook checked".to_string())
            } else {
                (StatusCode::BAD_REQUEST, "Webhook check error".to_string())
            }
        }
        Err(_) => (StatusCode::BAD_REQUEST, "Webhook check error".to_string()),
    }
}

async fn handle_update_webhook(state: &AppState) -> (StatusCode, String) {
    let Some(token) = state.telegram_token.as_deref() else {
        return (StatusCode::BAD_REQUEST, "Webhook update error".to_string());
    };
    let Some(function_url) = state.function_url.as_deref() else {
        return (StatusCode::BAD_REQUEST, "Webhook update error".to_string());
    };
    let Some(webhook_key) = state.webhook_key.as_deref() else {
        return (StatusCode::BAD_REQUEST, "Webhook update error".to_string());
    };

    let mut redis = match state.redis.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(_) => return (StatusCode::BAD_REQUEST, "Webhook update error".to_string()),
    };

    match telegram::set_webhook(&state.http, token, function_url, webhook_key, &mut redis)
        .await
    {
        Ok(true) => (StatusCode::OK, "Webhook updated".to_string()),
        _ => (StatusCode::BAD_REQUEST, "Webhook update error".to_string()),
    }
}

async fn is_secret_token_valid(state: &AppState, headers: &HeaderMap) -> bool {
    let Some(secret_header) = headers
        .get("X-Telegram-Bot-Api-Secret-Token")
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };

    let mut redis = match state.redis.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(_) => return false,
    };

    let stored: Option<String> =
        match redis_get_string(&mut redis, "X-Telegram-Bot-Api-Secret-Token").await {
            Ok(value) => value,
            Err(_) => None,
        };

    match stored {
        Some(value) => value == secret_header,
        None => false,
    }
}

fn is_true(value: &Option<String>) -> bool {
    matches!(value.as_deref(), Some("true") | Some("TRUE") | Some("True"))
}

const RATE_LIMIT_GLOBAL_MAX: i64 = 1024;
const RATE_LIMIT_CHAT_MAX: i64 = 128;
const TTL_RATE_GLOBAL: u64 = 3600;
const TTL_RATE_CHAT: u64 = 600;

async fn handle_message(state: &AppState, message: crate::models::Message) -> Result<(), String> {
    let Some(token) = state.telegram_token.as_deref() else {
        return Err("TELEGRAM_TOKEN not configured".to_string());
    };
    let chat_id = message.chat.id;
    let message_id = message.message_id;
    let text = message.text.clone().unwrap_or_default();

    if rate_limited(state, chat_id).await {
        let _ = telegram::send_message(
            &state.http,
            token,
            chat_id,
            "Estás mandando demasiados mensajes, bancá un toque.",
            None,
            None,
        )
        .await;
        return Ok(());
    }

    save_message_to_redis(state, chat_id, format!("user_{message_id}"), &text).await;

    if let Some((command, args)) = commands::parse_command(&text, state.bot_username.as_deref()) {
        if command == "/config" {
            let ctx = config_context(state);
            if let Ok((config_text, keyboard)) =
                chat_config::handle_config_command(&ctx, chat_id).await
            {
                let _ = telegram::send_message(
                    &state.http,
                    token,
                    chat_id,
                    &config_text,
                    Some(keyboard),
                    None,
                )
                .await;
            }
            return Ok(());
        }
        match command.as_str() {
            "/help" | "/start" => {
                let reply = commands::help_text();
                send_and_track(state, token, chat_id, &reply).await;
                return Ok(());
            }
            "/time" => {
                let reply = format!("{}", chrono::Utc::now().timestamp());
                send_and_track(state, token, chat_id, &reply).await;
                return Ok(());
            }
            "/search" | "/buscar" => {
                let results = tools::web_search(&state.http, &args, 5).await;
                let reply = tools::format_search_results(&args, &results);
                send_and_track(state, token, chat_id, &reply).await;
                return Ok(());
            }
            "/prices" | "/precio" | "/precios" | "/presio" | "/presios" => {
                if let Some(reply) = market::get_prices(&state.http, &state.redis, &args).await {
                    send_and_track(state, token, chat_id, &reply).await;
                }
                return Ok(());
            }
            "/dolar" | "/dollar" | "/usd" => {
                if let Some(reply) = market::get_dollar_rates(&state.http, &state.redis).await {
                    send_and_track(state, token, chat_id, &reply).await;
                }
                return Ok(());
            }
            "/bcra" | "/variables" => {
                if let Some(reply) = bcra::get_bcra_variables(&state.http, &state.redis).await {
                    send_and_track(state, token, chat_id, &reply).await;
                }
                return Ok(());
            }
            "/devo" => {
                let reply = market::get_devo(&state.http, &state.redis, &args).await;
                send_and_track(state, token, chat_id, &reply).await;
                return Ok(());
            }
            "/rulo" => {
                let reply = market::get_rulo(&state.http, &state.redis).await;
                send_and_track(state, token, chat_id, &reply).await;
                return Ok(());
            }
            "/powerlaw" => {
                let reply = market::powerlaw(&state.http, &state.redis).await;
                send_and_track(state, token, chat_id, &reply).await;
                return Ok(());
            }
            "/rainbow" => {
                let reply = market::rainbow(&state.http, &state.redis).await;
                send_and_track(state, token, chat_id, &reply).await;
                return Ok(());
            }
            "/convertbase" => {
                let reply = market::convert_base(&args);
                send_and_track(state, token, chat_id, &reply).await;
                return Ok(());
            }
            "/transcribe" => {
                let reply = handle_transcribe(state, token, &message).await;
                send_and_track(state, token, chat_id, &reply).await;
                return Ok(());
            }
            "/ask" | "/pregunta" | "/che" | "/gordo" => {
                let system_prompt = config::load_bot_config()
                    .map(|cfg| cfg.system_prompt)
                    .unwrap_or_else(|_| "Sos un asistente útil.".to_string());
                let time_label = chrono::Local::now().format("%A %d/%m/%Y").to_string();
                let market_info = market::get_market_context(&state.http, &state.redis).await;
                let weather_info = weather::get_weather_context(&state.http, &state.redis).await;
                let hn_items = hacker_news::get_hn_context(&state.http).await;
                let hn_info = if hn_items.is_empty() {
                    None
                } else {
                    Some(hacker_news::format_hn_items(&hn_items))
                };
                let agent_memory = agent::get_agent_memory(&state.redis, 5).await;
                let context = ai::AiContext {
                    system_prompt,
                    time_label,
                    market_info,
                    weather_info,
                    hn_info,
                    agent_memory,
                };
                let user_message = ai::ChatMessage { role: "user".to_string(), content: args };
                if let Some(reply) = ai::ask_ai(&state.http, &context, vec![user_message]).await {
                    send_and_track(state, token, chat_id, &reply).await;
                }
                return Ok(());
            }
            _ => {}
        }
    }

    Ok(())
}

async fn handle_transcribe(
    state: &AppState,
    token: &str,
    message: &crate::models::Message,
) -> String {
    let Some(reply) = message.reply_to_message.as_ref() else {
        return "Respondé a un mensaje con audio o imagen para transcribir/describir".to_string();
    };

    if let Some(audio) = reply.audio.as_ref().or(reply.voice.as_ref()) {
        match media::transcribe_file_by_id(&state.http, &state.redis, token, &audio.file_id, true)
            .await
        {
            Ok(Some(text)) => return format!("🎵 Transcripción: {}", text),
            Err(code) if code == "download" => return "No pude descargar el audio".to_string(),
            _ => return "No pude transcribir el audio, intentá más tarde".to_string(),
        }
    }

    if let Some(photo) = reply.photo.as_ref().and_then(|photos| photos.last()) {
        match media::describe_media_by_id(
            &state.http,
            &state.redis,
            token,
            &photo.file_id,
            "Describe what you see in this image in detail.",
        )
        .await
        {
            Ok(Some(text)) => return format!("🖼️ Descripción: {}", text),
            Err(code) if code == "download" => return "No pude descargar la imagen".to_string(),
            _ => return "No pude describir la imagen, intentá más tarde".to_string(),
        }
    }

    if let Some(sticker) = reply.sticker.as_ref() {
        match media::describe_media_by_id(
            &state.http,
            &state.redis,
            token,
            &sticker.file_id,
            "Describe what you see in this sticker in detail.",
        )
        .await
        {
            Ok(Some(text)) => return format!("🎨 Descripción del sticker: {}", text),
            Err(code) if code == "download" => return "No pude descargar el sticker".to_string(),
            _ => return "No pude describir el sticker, intentá más tarde".to_string(),
        }
    }

    "El mensaje no contiene audio, imagen o sticker para transcribir/describir".to_string()
}

async fn send_and_track(
    state: &AppState,
    token: &str,
    chat_id: i64,
    reply: &str,
) {
    let response_id = telegram::send_message(&state.http, token, chat_id, reply, None, None)
        .await
        .ok()
        .flatten();
    if let Some(bot_id) = response_id {
        save_message_to_redis(state, chat_id, format!("bot_{bot_id}"), reply).await;
    }
}

async fn rate_limited(state: &AppState, chat_id: i64) -> bool {
    let global_key = "rate:global";
    let chat_key = format!("rate:chat:{chat_id}");

    let global_count = chat_config::increment_rate_limit(
        &state.redis,
        global_key,
        TTL_RATE_GLOBAL,
    )
    .await;
    let chat_count =
        chat_config::increment_rate_limit(&state.redis, &chat_key, TTL_RATE_CHAT).await;

    global_count > RATE_LIMIT_GLOBAL_MAX || chat_count > RATE_LIMIT_CHAT_MAX
}

async fn save_message_to_redis(
    state: &AppState,
    chat_id: i64,
    message_id: String,
    text: &str,
) {
    let mut redis = match state.redis.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(_) => return,
    };

    let history_key = format!("chat_history:{chat_id}");
    let message_ids_key = format!("chat_message_ids:{chat_id}");

    let exists: bool = redis::AsyncCommands::sismember(&mut redis, &message_ids_key, &message_id)
        .await
        .unwrap_or(false);
    if exists {
        return;
    }

    let payload = serde_json::json!({
        "id": message_id,
        "text": text,
        "timestamp": chrono::Utc::now().timestamp(),
    });

    let entry = payload.to_string();
    let _: () = redis::AsyncCommands::lpush(&mut redis, &history_key, entry)
        .await
        .unwrap_or(());
    let _: () = redis::AsyncCommands::sadd(&mut redis, &message_ids_key, &message_id)
        .await
        .unwrap_or(());
    let _: () = redis::AsyncCommands::ltrim(&mut redis, &history_key, 0, 31)
        .await
        .unwrap_or(());
}

async fn admin_report(state: &AppState, message: &str) {
    let Some(token) = state.telegram_token.as_deref() else {
        return;
    };
    let Some(chat_id) = state.admin_chat_id else {
        return;
    };

    let _ = telegram::send_message(&state.http, token, chat_id, message, None, None).await;
}

fn config_context(state: &AppState) -> chat_config::ConfigContext<'_> {
    chat_config::ConfigContext {
        http: &state.http,
        redis: &state.redis,
        token: state.telegram_token.as_deref(),
        webhook_key: state.webhook_key.as_deref(),
        function_url: state.function_url.as_deref(),
        admin_chat_id: state.admin_chat_id,
    }
}
