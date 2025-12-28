use chrono::{DateTime, FixedOffset, Local};
use redis::AsyncCommands;
use serde_json::{json, Value};

use crate::{ai, config, hacker_news};

const AGENT_THOUGHTS_KEY: &str = "agent:thoughts";
const MAX_AGENT_THOUGHTS: usize = 10;
const AGENT_THOUGHT_CHAR_LIMIT: usize = 500;
const AGENT_RECENT_THOUGHT_WINDOW: usize = 5;
const AGENT_EMPTY_RESPONSE_FALLBACK: &str = "HALLAZGOS: no se me ocurrio nada nuevo, pintó el vacío.\nPRÓXIMO PASO: meter una busqueda puntual para traer un dato real y salir de la fiaca.";

#[derive(Debug, Clone)]
pub struct AgentThought {
    pub text: String,
    pub timestamp: Option<i64>,
}

pub async fn get_agent_memory(redis: &redis::Client, limit: usize) -> Option<String> {
    let thoughts = load_agent_thoughts(redis, limit).await;
    if thoughts.is_empty() {
        return None;
    }

    let mut lines = Vec::new();
    let tz = FixedOffset::west_opt(3 * 3600).unwrap_or_else(|| FixedOffset::east_opt(0).unwrap());

    for thought in thoughts {
        if thought.text.is_empty() {
            continue;
        }
        if let Some(ts) = thought.timestamp {
            if let Some(dt) = DateTime::from_timestamp(ts, 0) {
                let local = dt.with_timezone(&tz);
                lines.push(format!("- [{}] {}", local.format("%d/%m %H:%M"), thought.text));
            } else {
                lines.push(format!("- {}", thought.text));
            }
        } else {
            lines.push(format!("- {}", thought.text));
        }
    }

    if lines.is_empty() {
        None
    } else {
        Some(format!(
            "MEMORIA AUTÓNOMA (más reciente primero):\n{}",
            lines.join("\n")
        ))
    }
}

pub async fn run_agent_cycle(
    http: &reqwest::Client,
    redis: &redis::Client,
) -> Result<Value, String> {
    let thoughts = load_agent_thoughts(redis, AGENT_RECENT_THOUGHT_WINDOW).await;
    let last_text = thoughts.first().map(|t| t.text.clone());
    let hn_items = hacker_news::get_hn_context(http).await;
    let hn_info = if hn_items.is_empty() {
        None
    } else {
        Some(hacker_news::format_hn_items(&hn_items))
    };

    let mut prompt = String::from(
        "Estas operando en modo autonomo. Podes investigar, navegar y usar herramientas. \
Registrá en primera persona que investigaste, que encontraste y recien despues el proximo paso. \
Devolve la nota en dos secciones en mayusculas: \"HALLAZGOS:\" y \"PROXIMO PASO:\".",
    );

    if let Some(text) = &last_text {
        if !text.is_empty() {
            prompt.push_str("\n\nULTIMA MEMORIA GUARDADA:\n");
            prompt.push_str(&truncate_text(text, 220));
            prompt.push_str("\nResolvé ese pendiente ahora mismo y deja asentado el resultado antes de planear otra cosa.");
        }
    }

    if let Some(hn_block) = &hn_info {
        prompt.push_str("\n\nHACKER NEWS HOY:\n");
        prompt.push_str(hn_block);
        prompt.push_str("\nSi alguna nota trae datos frescos que sumen, citá la fuente y metela en los hallazgos.");
    }

    prompt.push_str(
        "\nIncluí datos específicos (numeros, titulares, fuentes) de lo que investigues y evitá repetir entradas previas. \
Si necesitás info fresca, llamá a la herramienta web_search con un query puntual y resumí el hallazgo. \
Si hace falta leer una nota puntual, llamá a fetch_url con la URL (incluí https://) y anotá lo relevante. \
Maximo 500 caracteres, sin saludar a nadie: es un apunte privado.",
    );

    let system_prompt = config::load_bot_config()
        .map(|cfg| cfg.system_prompt)
        .unwrap_or_else(|_| "Sos un asistente útil.".to_string());
    let time_label = Local::now().format("%A %d/%m/%Y").to_string();
    let agent_memory = get_agent_memory(redis, AGENT_RECENT_THOUGHT_WINDOW).await;

    let context = ai::AiContext {
        system_prompt,
        time_label,
        market_info: None,
        weather_info: None,
        hn_info,
        agent_memory,
    };

    let user_message = ai::ChatMessage {
        role: "user".to_string(),
        content: prompt,
    };

    let reply = ai::ask_ai(http, redis, &context, vec![user_message])
        .await
        .ok_or_else(|| "Autonomous agent execution failed".to_string())?;

    let mut cleaned = sanitize_tool_artifacts(&reply);
    cleaned = cleaned.trim().to_string();
    if cleaned.is_empty() {
        return Ok(json!({"text": AGENT_EMPTY_RESPONSE_FALLBACK, "persisted": false}));
    }

    if let Some(last) = &last_text {
        if normalize_text(&cleaned) == normalize_text(last) {
            cleaned = AGENT_EMPTY_RESPONSE_FALLBACK.to_string();
        }
    }

    if !has_required_sections(&cleaned) {
        cleaned = AGENT_EMPTY_RESPONSE_FALLBACK.to_string();
    }

    let cleaned = truncate_text(&cleaned, AGENT_THOUGHT_CHAR_LIMIT);
    let entry = save_agent_thought(redis, &cleaned)
        .await
        .ok_or_else(|| "Failed to persist autonomous agent thought".to_string())?;

    let mut result = json!({
        "text": entry.text,
        "persisted": true,
    });

    if let Some(ts) = entry.timestamp {
        if let Some(dt) = DateTime::from_timestamp(ts, 0) {
            let tz = FixedOffset::west_opt(3 * 3600)
                .unwrap_or_else(|| FixedOffset::east_opt(0).unwrap());
            result["timestamp"] = json!(ts);
            result["iso_time"] = json!(dt.with_timezone(&tz).to_rfc3339());
        }
    }

    Ok(result)
}

async fn load_agent_thoughts(
    redis: &redis::Client,
    limit: usize,
) -> Vec<AgentThought> {
    let mut conn = match redis.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(_) => return Vec::new(),
    };

    let raw: Vec<String> = match conn
        .lrange(AGENT_THOUGHTS_KEY, 0, (limit.saturating_sub(1)) as isize)
        .await
    {
        Ok(items) => items,
        Err(_) => return Vec::new(),
    };

    raw.into_iter().filter_map(parse_thought).collect()
}

async fn save_agent_thought(
    redis: &redis::Client,
    text: &str,
) -> Option<AgentThought> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    let text = truncate_text(trimmed, AGENT_THOUGHT_CHAR_LIMIT);
    let timestamp = chrono::Utc::now().timestamp();
    let payload = json!({
        "text": text,
        "timestamp": timestamp,
    });

    let mut conn = redis.get_multiplexed_async_connection().await.ok()?;
    let _: () = conn.lpush(AGENT_THOUGHTS_KEY, payload.to_string()).await.ok()?;
    let _: () = conn
        .ltrim(AGENT_THOUGHTS_KEY, 0, (MAX_AGENT_THOUGHTS - 1) as isize)
        .await
        .ok()?;

    Some(AgentThought {
        text,
        timestamp: Some(timestamp),
    })
}

fn parse_thought(raw: String) -> Option<AgentThought> {
    let value = serde_json::from_str::<Value>(&raw).ok()?;
    let text = value
        .get("text")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    if text.is_empty() {
        return None;
    }
    let timestamp = parse_timestamp(value.get("timestamp"));
    Some(AgentThought { text, timestamp })
}

fn parse_timestamp(value: Option<&Value>) -> Option<i64> {
    match value {
        Some(Value::Number(num)) => num.as_i64(),
        Some(Value::String(s)) => s.parse::<i64>().ok(),
        _ => None,
    }
}

fn normalize_text(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn has_required_sections(text: &str) -> bool {
    let upper = text.to_uppercase();
    upper.contains("HALLAZGOS") && (upper.contains("PRÓXIMO PASO") || upper.contains("PROXIMO PASO"))
}

fn sanitize_tool_artifacts(text: &str) -> String {
    if let Some(pos) = text.find("[TOOL]") {
        return text[..pos].trim().to_string();
    }
    text.trim().to_string()
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (idx, ch) in text.chars().enumerate() {
        if idx >= max_chars {
            break;
        }
        out.push(ch);
    }
    out
}
