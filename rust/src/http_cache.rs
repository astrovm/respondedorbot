use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::redis_store::redis_get_string;
use crate::redis_store::redis_set_string;

pub async fn cached_get_json(
    http: &reqwest::Client,
    redis: &redis::Client,
    url: &str,
    params: Option<&[(&str, String)]>,
    headers: Option<&[(&str, String)]>,
    ttl_seconds: u64,
) -> Option<serde_json::Value> {
    let hash_key = build_cache_key(url, params, headers);
    let cache_key = format!("cache:{hash_key}");

    let now = current_timestamp();
    if let Ok(Some(raw)) = fetch_cached(redis, &cache_key).await {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) {
            if let Some(ts) = value.get("timestamp").and_then(|v| v.as_i64()) {
                if now - ts <= ttl_seconds as i64 {
                    if let Some(data) = value.get("data") {
                        return Some(data.clone());
                    }
                }
            }
        }
    }

    let response = http
        .get(url)
        .headers(build_headers(headers))
        .query(&params.unwrap_or_default())
        .send()
        .await
        .ok()?;

    let data = response.json::<serde_json::Value>().await.ok()?;
    let payload = json!({
        "timestamp": now,
        "data": data,
    });

    let mut conn = redis.get_multiplexed_async_connection().await.ok()?;
    let _ = redis_set_string(&mut conn, &cache_key, &payload.to_string()).await;

    payload.get("data").cloned()
}

fn build_cache_key(
    url: &str,
    params: Option<&[(&str, String)]>,
    headers: Option<&[(&str, String)]>,
) -> String {
    let mut parts = vec![url.to_string()];
    if let Some(params) = params {
        let mut sorted = params.to_vec();
        sorted.sort_by(|a, b| a.0.cmp(b.0));
        for (k, v) in sorted {
            parts.push(format!("{k}={v}"));
        }
    }
    if let Some(headers) = headers {
        let mut sorted = headers.to_vec();
        sorted.sort_by(|a, b| a.0.cmp(b.0));
        for (k, v) in sorted {
            parts.push(format!("{k}={v}"));
        }
    }
    let joined = parts.join("|");
    let digest = md5::compute(joined.as_bytes());
    format!("{:x}", digest)
}

fn build_headers(headers: Option<&[(&str, String)]>) -> reqwest::header::HeaderMap {
    let mut map = reqwest::header::HeaderMap::new();
    if let Some(headers) = headers {
        for (key, value) in headers {
            if let (Ok(name), Ok(val)) = (
                reqwest::header::HeaderName::from_bytes(key.as_bytes()),
                reqwest::header::HeaderValue::from_str(value),
            ) {
                map.insert(name, val);
            }
        }
    }
    map
}

async fn fetch_cached(
    redis: &redis::Client,
    key: &str,
) -> redis::RedisResult<Option<String>> {
    let mut conn = redis.get_multiplexed_async_connection().await?;
    redis_get_string(&mut conn, key).await
}

fn current_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}
