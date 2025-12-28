use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::redis_store::redis_get_string;
use crate::redis_store::redis_set_string;

pub struct CacheOptions {
    pub ttl_seconds: u64,
    pub hourly_cache: bool,
    pub history_hours_ago: Option<u32>,
}

pub async fn cached_get_json(
    http: &reqwest::Client,
    redis: &redis::Client,
    url: &str,
    params: Option<&[(&str, String)]>,
    headers: Option<&[(&str, String)]>,
    ttl_seconds: u64,
) -> Option<serde_json::Value> {
    cached_get_json_full(
        http,
        redis,
        url,
        params,
        headers,
        CacheOptions {
            ttl_seconds,
            hourly_cache: false,
            history_hours_ago: None,
        },
    )
        .await
        .and_then(|value| value.get("data").cloned())
}

pub async fn cached_get_json_full(
    http: &reqwest::Client,
    redis: &redis::Client,
    url: &str,
    params: Option<&[(&str, String)]>,
    headers: Option<&[(&str, String)]>,
    options: CacheOptions,
) -> Option<serde_json::Value> {
    let hash_key = build_cache_key(url, params, headers);
    let cache_key = format!("cache:{hash_key}");
    let now = current_timestamp();

    let cached = fetch_cached(redis, &cache_key).await.ok().flatten();
    let cache_history = if let Some(hours) = options.history_hours_ago {
        get_cache_history(hours, &hash_key, redis).await
    } else {
        None
    };

    if let Some(raw) = cached.as_ref() {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) {
            if let Some(ts) = value.get("timestamp").and_then(|v| v.as_i64()) {
                if now - ts <= options.ttl_seconds as i64 {
                    let mut out = value.clone();
                    if let Some(history) = cache_history {
                        out["history"] = history;
                    }
                    return Some(out);
                }
            }
        }
    }

    let response = match make_request(http, url, params, headers).await {
        Ok(resp) => resp,
        Err(_) => {
            if let Some(raw) = cached {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) {
                    return Some(value);
                }
            }
            return None;
        }
    };

    let data = response.json::<serde_json::Value>().await.ok()?;
    let mut payload = json!({
        "timestamp": now,
        "data": data,
    });
    if options.hourly_cache {
        let hour_key = hourly_cache_key(now, &hash_key);
        let mut conn = redis.get_multiplexed_async_connection().await.ok()?;
        let _ = redis_set_string(&mut conn, &hour_key, &payload.to_string()).await;
    }
    if let Some(history) = cache_history {
        payload["history"] = history;
    }

    let mut conn = redis.get_multiplexed_async_connection().await.ok()?;
    let _ = redis_set_string(&mut conn, &cache_key, &payload.to_string()).await;

    Some(payload)
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

fn hourly_cache_key(timestamp: i64, hash_key: &str) -> String {
    let dt = chrono::DateTime::<chrono::Utc>::from_timestamp(timestamp, 0)
        .unwrap_or_else(chrono::Utc::now);
    format!("{}{}", dt.format("%Y-%m-%d-%H"), hash_key)
}

pub async fn get_cache_history(
    hours_ago: u32,
    hash_key: &str,
    redis: &redis::Client,
) -> Option<serde_json::Value> {
    let now = chrono::Utc::now();
    let ts = now - chrono::Duration::hours(hours_ago as i64);
    let key = format!("{}{}", ts.format("%Y-%m-%d-%H"), hash_key);
    let mut conn = redis.get_multiplexed_async_connection().await.ok()?;
    let raw = redis_get_string(&mut conn, &key).await.ok().flatten()?;
    let parsed = serde_json::from_str::<serde_json::Value>(&raw).ok()?;
    if parsed.get("timestamp").is_some() {
        Some(parsed)
    } else {
        None
    }
}

async fn make_request(
    http: &reqwest::Client,
    url: &str,
    params: Option<&[(&str, String)]>,
    headers: Option<&[(&str, String)]>,
) -> Result<reqwest::Response, reqwest::Error> {
    let request = http
        .get(url)
        .headers(build_headers(headers))
        .query(&params.unwrap_or_default());
    match request.send().await {
        Ok(resp) => Ok(resp),
        Err(_) => crate::http::get_with_ssl_fallback(url).await,
    }
}
