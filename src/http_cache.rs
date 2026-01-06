use http::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::http::{self as http_client, HttpClient, HttpResponse};
use crate::storage::Storage;

pub struct CacheOptions {
    pub ttl_seconds: u64,
    pub hourly_cache: bool,
    pub history_hours_ago: Option<u32>,
}

pub async fn cached_get_json(
    http: &HttpClient,
    storage: &Storage,
    url: &str,
    params: Option<&[(&str, String)]>,
    headers: Option<&[(&str, String)]>,
    ttl_seconds: u64,
) -> Option<serde_json::Value> {
    cached_get_json_full(
        http,
        storage,
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
    http: &HttpClient,
    storage: &Storage,
    url: &str,
    params: Option<&[(&str, String)]>,
    headers: Option<&[(&str, String)]>,
    options: CacheOptions,
) -> Option<serde_json::Value> {
    let hash_key = build_cache_key(url, params, headers);
    let cache_key = format!("cache:{hash_key}");
    let now = current_timestamp();

    let cached = storage.get_string(&cache_key).await;
    let cache_history = if let Some(hours) = options.history_hours_ago {
        get_cache_history(hours, &hash_key, storage).await
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
        let _ = storage.set_string(&hour_key, &payload.to_string()).await;
    }
    if let Some(history) = cache_history {
        payload["history"] = history;
    }

    let _ = storage.set_string(&cache_key, &payload.to_string()).await;

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

fn build_headers(headers: Option<&[(&str, String)]>) -> HeaderMap {
    let mut map = HeaderMap::new();
    if let Some(headers) = headers {
        for (key, value) in headers {
            if let (Ok(name), Ok(val)) = (
                HeaderName::from_bytes(key.as_bytes()),
                HeaderValue::from_str(value),
            ) {
                map.insert(name, val);
            }
        }
    }
    map
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
    storage: &Storage,
) -> Option<serde_json::Value> {
    let now = chrono::Utc::now();
    let ts = now - chrono::Duration::hours(hours_ago as i64);
    let key = format!("{}{}", ts.format("%Y-%m-%d-%H"), hash_key);
    let raw = storage.get_string(&key).await?;
    let parsed = serde_json::from_str::<serde_json::Value>(&raw).ok()?;
    if parsed.get("timestamp").is_some() {
        Some(parsed)
    } else {
        None
    }
}

async fn make_request(
    http: &HttpClient,
    url: &str,
    params: Option<&[(&str, String)]>,
    headers: Option<&[(&str, String)]>,
) -> Result<HttpResponse, http_client::HttpError> {
    let mut request = http.get(url).headers(build_headers(headers));
    if let Some(params) = params {
        for (key, value) in params {
            request = request.query(*key, value.clone());
        }
    }
    match request.send().await {
        Ok(resp) => Ok(resp),
        Err(_) => http_client::get_with_ssl_fallback(http, url).await,
    }
}
