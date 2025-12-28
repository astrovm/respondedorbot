use chrono::{DateTime, FixedOffset};
use redis::AsyncCommands;
use serde_json::Value;

const AGENT_THOUGHTS_KEY: &str = "agent:thoughts";

pub async fn get_agent_memory(redis: &redis::Client, limit: usize) -> Option<String> {
    let mut conn = redis.get_multiplexed_async_connection().await.ok()?;
    let raw: Vec<String> = conn
        .lrange(AGENT_THOUGHTS_KEY, 0, (limit.saturating_sub(1)) as isize)
        .await
        .ok()?;

    if raw.is_empty() {
        return None;
    }

    let mut lines = Vec::new();
    let tz = FixedOffset::west_opt(3 * 3600).unwrap_or_else(|| FixedOffset::east_opt(0).unwrap());

    for item in raw {
        let parsed = serde_json::from_str::<Value>(&item).ok();
        let text = parsed
            .as_ref()
            .and_then(|v| v.get("text"))
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        if text.is_empty() {
            continue;
        }
        let timestamp = parsed
            .as_ref()
            .and_then(|v| v.get("timestamp"))
            .and_then(|v| v.as_i64());
        if let Some(ts) = timestamp {
            if let Some(dt) = DateTime::from_timestamp(ts, 0) {
                let local = dt.with_timezone(&tz);
                lines.push(format!("- [{}] {}", local.format("%d/%m %H:%M"), text));
            } else {
                lines.push(format!("- {}", text));
            }
        } else {
            lines.push(format!("- {}", text));
        }
    }

    if lines.is_empty() {
        None
    } else {
        Some(format!("MEMORIA AUTÓNOMA (más reciente primero):\n{}", lines.join("\n")))
    }
}
