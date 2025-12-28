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
mod http;
mod http_cache;
mod market;
mod ai;
mod agent;
mod bcra;
mod polymarket;
mod tools;
mod weather;
mod hacker_news;
mod media;
mod links;
mod message_utils;
mod models;

use crate::message_utils::{
    build_ai_messages, build_link_keyboard, build_reply_context_text, contains_url,
    format_shared_by, format_user_identity, format_user_message, gen_random, is_group_chat,
    post_process_ai_response, should_gordo_respond_core, truncate_text, ChatHistoryEntry,
};
use crate::models::Update;
use crate::redis_store::{create_redis_client, redis_get_string, redis_setex_string};

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
        let updated = market::get_dollar_rates(&state.http, &state.redis)
            .await
            .is_some();
        if updated {
            return (StatusCode::OK, "Dollars updated".to_string());
        }
        return (StatusCode::BAD_REQUEST, "Dollars update error".to_string());
    }

    if is_true(&query.run_agent) {
        match agent::run_agent_cycle(&state.http, &state.redis).await {
            Ok(payload) => return (StatusCode::OK, payload.to_string()),
            Err(err) => {
                tracing::error!(error = %err, "run_agent failed");
                admin_report(&state, "Agent run failed").await;
                return (StatusCode::INTERNAL_SERVER_ERROR, "Agent run failed".to_string());
            }
        }
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
        redis_get_string(&mut redis, "X-Telegram-Bot-Api-Secret-Token")
            .await
            .unwrap_or_default();

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
const BOT_MESSAGE_META_PREFIX: &str = "bot_message_meta:";
const BOT_MESSAGE_META_TTL: u64 = 3 * 24 * 60 * 60;

async fn handle_message(state: &AppState, message: crate::models::Message) -> Result<(), String> {
    let Some(token) = state.telegram_token.as_deref() else {
        return Err("TELEGRAM_TOKEN not configured".to_string());
    };
    let chat_id = message.chat.id;
    let message_id = message.message_id;
    let mut text = message.text.clone().unwrap_or_default();
    let user_id = message.from.as_ref().map(|u| u.id).unwrap_or(0);
    let username = message
        .from
        .as_ref()
        .and_then(|u| u.username.as_deref())
        .unwrap_or("");
    tracing::info!(chat_id, message_id, user_id, username, "message received");

    let chat_config = chat_config::get_chat_config(&state.redis, chat_id).await;
    let user_identity = format_user_identity(message.from.as_ref());
    let user_identity_opt = if user_identity.trim().is_empty() {
        None
    } else {
        Some(user_identity.as_str())
    };

    let is_transcribe_command = text.trim().to_lowercase().starts_with("/transcribe");
    if !is_transcribe_command {
        if let Some(audio) = message.audio.as_ref().or(message.voice.as_ref()) {
            match media::transcribe_file_by_id(
                &state.http,
                &state.redis,
                token,
                &audio.file_id,
                false,
            )
            .await
            {
                Ok(Some(transcription)) => {
                    text = transcription;
                }
                Err(code) if code == "download" => {
                    text = "no pude bajar tu audio, mandalo de vuelta".to_string();
                }
                Err(_) => {
                    text = "mandame texto que no soy alexa, boludo".to_string();
                }
                _ => {}
            }
        }
    }

    if text.trim().is_empty() && message.photo.as_ref().is_some() {
        text = "que onda con esta foto".to_string();
    }

    if !text.is_empty() && chat_config.link_mode != "off" && !text.trim().starts_with('/') {
        let (rewritten, changed, original_links) = links::replace_links(&state.http, &text).await;
        if changed {
            tracing::info!(chat_id, message_id, "link rewrite applied");
            let mut final_text = rewritten;
            if let Some(shared_by) = format_shared_by(message.from.as_ref()) {
                if !shared_by.is_empty() {
                    final_text.push_str(&format!("\n\nShared by {shared_by}"));
                }
            }
            let reply_to = if chat_config.link_mode == "reply" {
                message
                    .reply_to_message
                    .as_ref()
                    .map(|msg| msg.message_id)
                    .or(Some(message_id))
            } else {
                message.reply_to_message.as_ref().map(|msg| msg.message_id)
            };
            let keyboard = if original_links.is_empty() {
                None
            } else {
                Some(build_link_keyboard(&original_links))
            };
            let response_id = telegram::send_message(
                &state.http,
                token,
                chat_id,
                &final_text,
                keyboard,
                reply_to,
            )
            .await
            .ok()
            .flatten();
            if let Some(bot_id) = response_id {
                track_bot_response(
                    state,
                    chat_id,
                    bot_id,
                    &final_text,
                    Some(serde_json::json!({"type": "link_fix"})),
                )
                .await;
            }
            if chat_config.link_mode == "delete" {
                let _ = telegram::delete_message(&state.http, token, chat_id, message_id).await;
            }
            return Ok(());
        }
        if contains_url(&text) {
            return Ok(());
        }
    }

    let reply_context_text = build_reply_context_text(&message);
    let reply_metadata = load_reply_metadata(state, &message).await;

    let mut response_msg: Option<String> = None;
    let mut response_command: Option<String> = None;
    let mut response_uses_ai = false;
    let mut response_markup: Option<serde_json::Value> = None;

    if let Some((command, args)) = commands::parse_command(&text, state.bot_username.as_deref()) {
        tracing::info!(chat_id, message_id, command = %command, "command received");
        response_command = Some(command.clone());

        if command == "/config" && is_group_chat(&message) {
            let Some(user_id) = message.from.as_ref().map(|u| u.id) else {
                return Ok(());
            };
            let ctx = config_context(state);
            let is_admin = chat_config::is_chat_admin(&ctx, chat_id, user_id).await;
            if !is_admin {
                let denial = "Solo los admins pueden cambiar la config del gordo acá.";
                let _ = telegram::send_message(
                    &state.http,
                    token,
                    chat_id,
                    denial,
                    None,
                    Some(message_id),
                )
                .await;
                chat_config::report_unauthorized(
                    &ctx,
                    chat_id,
                    message.chat.kind.as_str(),
                    message.from.as_ref().and_then(|u| u.username.as_deref()),
                )
                .await;
                return Ok(());
            }
        }

        if command == "/config" {
            let ctx = config_context(state);
            if let Ok((config_text, keyboard)) =
                chat_config::handle_config_command(&ctx, chat_id).await
            {
                response_msg = Some(config_text);
                response_markup = Some(keyboard);
            }
        } else if command == "/help" || command == "/start" {
            response_msg = Some(commands::help_text());
        } else if command == "/time" {
            response_msg = Some(format!("{}", chrono::Utc::now().timestamp()));
        } else if command == "/instance" {
            let instance = std::env::var("FRIENDLY_INSTANCE_NAME").unwrap_or_default();
            response_msg = Some(format!("estoy corriendo en {} boludo", instance));
        } else if command == "/search" || command == "/buscar" {
            let results = tools::web_search(&state.http, &state.redis, &args, 5).await;
            response_msg = Some(tools::format_search_results(&args, &results));
        } else if matches!(command.as_str(), "/prices" | "/precio" | "/precios" | "/presio" | "/presios") {
            response_msg = market::get_prices(&state.http, &state.redis, &args).await;
        } else if matches!(command.as_str(), "/dolar" | "/dollar" | "/usd") {
            response_msg = market::get_dollar_rates(&state.http, &state.redis).await;
        } else if matches!(command.as_str(), "/bcra" | "/variables") {
            response_msg = bcra::get_bcra_variables(&state.http, &state.redis).await;
        } else if command == "/eleccion" {
            response_msg = Some(
                polymarket::get_polymarket_argentina_election(&state.http, &state.redis).await,
            );
        } else if command == "/devo" {
            response_msg = Some(market::get_devo(&state.http, &state.redis, &args).await);
        } else if command == "/rulo" {
            response_msg = Some(market::get_rulo(&state.http, &state.redis).await);
        } else if matches!(command.as_str(), "/satoshi" | "/sat" | "/sats") {
            response_msg = Some(market::satoshi(&state.http, &state.redis).await);
        } else if command == "/random" {
            response_msg = Some(commands::select_random(&args));
        } else if matches!(command.as_str(), "/comando" | "/command") {
            response_msg = Some(commands::convert_to_command(&args));
        } else if command == "/powerlaw" {
            response_msg = Some(market::powerlaw(&state.http, &state.redis).await);
        } else if command == "/rainbow" {
            response_msg = Some(market::rainbow(&state.http, &state.redis).await);
        } else if command == "/convertbase" {
            response_msg = Some(market::convert_base(&args));
        } else if command == "/agent" {
            response_msg = Some(
                agent::get_agent_memory(&state.redis, 5)
                    .await
                    .unwrap_or_else(|| {
                        "todavía no tengo pensamientos guardados, dejame que labure un toque."
                            .to_string()
                    }),
            );
        } else if command == "/transcribe" {
            tracing::info!(chat_id, message_id, "transcribe command");
            response_msg = Some(handle_transcribe(state, token, &message).await);
        } else if matches!(command.as_str(), "/ask" | "/pregunta" | "/che" | "/gordo") {
            tracing::info!(chat_id, message_id, "ask command");
            if rate_limited(state, chat_id).await {
                response_msg = Some(handle_rate_limit(state, token, &message).await);
            } else {
                response_uses_ai = true;
                response_msg = run_ai_with_context(
                    state,
                    token,
                    &message,
                    &args,
                    reply_context_text.as_deref(),
                    user_identity_opt,
                )
                .await;
            }
        }
    } else {
        let should_respond = should_gordo_respond(
            state,
            &message,
            &text,
            &chat_config,
            reply_metadata.as_ref(),
        )
        .await;

        if should_respond {
            if rate_limited(state, chat_id).await {
                response_msg = Some(handle_rate_limit(state, token, &message).await);
            } else {
                response_uses_ai = true;
                response_msg = run_ai_with_context(
                    state,
                    token,
                    &message,
                    &text,
                    reply_context_text.as_deref(),
                    user_identity_opt,
                )
                .await;
            }
        }
    }

    if let Some(reply) = response_msg {
        if let Some(markup) = response_markup {
            let response_id = telegram::send_message(
                &state.http,
                token,
                chat_id,
                &reply,
                Some(markup),
                Some(message_id),
            )
            .await
            .ok()
            .flatten();
            if let Some(bot_id) = response_id {
                let metadata = response_command.as_ref().map(|cmd| {
                    serde_json::json!({"type": "command", "command": cmd, "uses_ai": response_uses_ai})
                }).unwrap_or_else(|| serde_json::json!({"type": "ai"}));
                track_bot_response(state, chat_id, bot_id, &reply, Some(metadata)).await;
            }
        } else {
            let metadata = response_command.as_ref().map(|cmd| {
                serde_json::json!({"type": "command", "command": cmd, "uses_ai": response_uses_ai})
            }).unwrap_or_else(|| serde_json::json!({"type": "ai"}));
            send_and_track(state, token, chat_id, &reply, Some(metadata), Some(message_id)).await;
        }

        if !text.is_empty() {
            let formatted = format_user_message(&message, &text, reply_context_text.as_deref());
            save_message_to_redis(state, chat_id, format!("user_{message_id}"), &formatted).await;
        }
        return Ok(());
    }

    if !text.is_empty() {
        let formatted = format_user_message(&message, &text, reply_context_text.as_deref());
        save_message_to_redis(state, chat_id, format!("user_{message_id}"), &formatted).await;
    }

    Ok(())
}

async fn run_ai_with_context(
    state: &AppState,
    token: &str,
    message: &crate::models::Message,
    message_text: &str,
    reply_context: Option<&str>,
    user_identity: Option<&str>,
) -> Option<String> {
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

    let mut enriched_text = message_text.to_string();
    if let Some(photo) = message.photo.as_ref().and_then(|list| list.last()) {
        if let Ok(Some(description)) = media::describe_media_by_id(
            &state.http,
            &state.redis,
            token,
            &photo.file_id,
            "Describe what you see in this image in detail.",
        )
        .await
        {
            let image_context = format!("[Imagen: {}]", description);
            if !enriched_text.trim().is_empty() {
                enriched_text.push_str("\n\n");
            }
            enriched_text.push_str(&image_context);
        }
    }

    if let Some(token) = state.telegram_token.as_deref() {
        let _ = telegram::send_chat_action(&state.http, token, message.chat.id, "typing").await;
    }
    tokio::time::sleep(std::time::Duration::from_millis(
        (rand::random::<f64>() * 1000.0) as u64,
    ))
    .await;

    let history = get_chat_history(&state.redis, message.chat.id, 8).await;
    let messages = build_ai_messages(message, &history, &enriched_text, reply_context);
    let raw = ai::ask_ai(&state.http, &state.redis, &context, messages).await?;
    let cleaned = post_process_ai_response(&raw, &[reply_context.map(|s| s.to_string())], user_identity);
    if cleaned.trim().is_empty() {
        Some("no pude generar respuesta, intentá de nuevo".to_string())
    } else {
        Some(cleaned)
    }
}

async fn get_chat_history(
    redis: &redis::Client,
    chat_id: i64,
    max_messages: usize,
) -> Vec<ChatHistoryEntry> {
    let mut conn = match redis.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(_) => return Vec::new(),
    };
    let history_key = format!("chat_history:{chat_id}");
    let raw: Vec<String> = match redis::AsyncCommands::lrange(
        &mut conn,
        &history_key,
        0,
        (max_messages.saturating_sub(1)) as isize,
    )
    .await
    {
        Ok(values) => values,
        Err(_) => return Vec::new(),
    };

    let mut entries = Vec::new();
    for entry in raw {
        let value = serde_json::from_str::<serde_json::Value>(&entry).ok();
        let text = value
            .as_ref()
            .and_then(|v| v.get("text"))
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        if text.is_empty() {
            continue;
        }
        let id = value
            .as_ref()
            .and_then(|v| v.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let role = if id.starts_with("bot_") {
            "assistant".to_string()
        } else {
            "user".to_string()
        };
        entries.push(ChatHistoryEntry { role, text });
    }
    entries.reverse();
    entries
}

async fn load_reply_metadata(
    state: &AppState,
    message: &crate::models::Message,
) -> Option<serde_json::Value> {
    let reply = message.reply_to_message.as_ref()?;
    let bot_username = state.bot_username.as_deref()?;
    let reply_user = reply.from.as_ref()?;
    if reply_user.username.as_deref() != Some(bot_username) {
        return None;
    }
    get_bot_message_metadata(&state.redis, message.chat.id, reply.message_id).await
}


async fn should_gordo_respond(
    state: &AppState,
    message: &crate::models::Message,
    text: &str,
    chat_config: &chat_config::ChatConfig,
    reply_metadata: Option<&serde_json::Value>,
) -> bool {
    let bot_username = state.bot_username.as_deref().unwrap_or("");
    let trigger_words = config::load_bot_config()
        .map(|cfg| cfg.trigger_words)
        .unwrap_or_else(|_| vec!["bot".to_string(), "assistant".to_string()]);
    let random_value = rand::random::<f64>();
    should_gordo_respond_core(
        bot_username,
        message,
        text,
        chat_config,
        reply_metadata,
        &trigger_words,
        random_value,
    )
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
    metadata: Option<serde_json::Value>,
    reply_to: Option<i64>,
) -> Option<i64> {
    let response_id = telegram::send_message(&state.http, token, chat_id, reply, None, reply_to)
        .await
        .ok()
        .flatten();
    if let Some(bot_id) = response_id {
        track_bot_response(state, chat_id, bot_id, reply, metadata).await;
        return Some(bot_id);
    }
    None
}

async fn track_bot_response(
    state: &AppState,
    chat_id: i64,
    message_id: i64,
    text: &str,
    metadata: Option<serde_json::Value>,
) {
    save_message_to_redis(state, chat_id, format!("bot_{message_id}"), text).await;
    if let Some(meta) = metadata {
        save_bot_message_metadata(&state.redis, chat_id, message_id, &meta).await;
    }
}

fn bot_message_meta_key(chat_id: i64, message_id: i64) -> String {
    format!("{BOT_MESSAGE_META_PREFIX}{chat_id}:{message_id}")
}

async fn save_bot_message_metadata(
    redis: &redis::Client,
    chat_id: i64,
    message_id: i64,
    metadata: &serde_json::Value,
) {
    let mut conn = match redis.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(_) => return,
    };
    let _ = redis_setex_string(
        &mut conn,
        &bot_message_meta_key(chat_id, message_id),
        BOT_MESSAGE_META_TTL,
        &metadata.to_string(),
    )
    .await;
}

async fn get_bot_message_metadata(
    redis: &redis::Client,
    chat_id: i64,
    message_id: i64,
) -> Option<serde_json::Value> {
    let mut conn = redis.get_multiplexed_async_connection().await.ok()?;
    let raw = redis_get_string(&mut conn, &bot_message_meta_key(chat_id, message_id))
        .await
        .ok()
        .flatten()?;
    serde_json::from_str::<serde_json::Value>(&raw).ok()
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

async fn handle_rate_limit(
    state: &AppState,
    token: &str,
    message: &crate::models::Message,
) -> String {
    let _ = telegram::send_chat_action(&state.http, token, message.chat.id, "typing").await;
    tokio::time::sleep(std::time::Duration::from_millis(
        (rand::random::<f64>() * 1000.0) as u64,
    ))
    .await;
    let name = message
        .from
        .as_ref()
        .map(|u| u.first_name.as_str())
        .unwrap_or("");
    gen_random(name)
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
        "text": truncate_text(text, 512),
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
