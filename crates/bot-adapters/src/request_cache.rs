//! Python-compatible cached JSON HTTP-request orchestration.

use bot_core::cache_policy::request_cache_ttl;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::redis_json_cache::{RedisJsonCache, RedisJsonCacheError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonHttpResponse {
    pub status_code: u16,
    pub body: String,
}

pub trait RequestCache {
    type Error: std::fmt::Display;

    fn get(&mut self, key: &str) -> Result<Option<String>, Self::Error>;

    fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error>;
}

impl RequestCache for RedisJsonCache {
    type Error = RedisJsonCacheError;

    fn get(&mut self, key: &str) -> Result<Option<String>, Self::Error> {
        RedisJsonCache::get(self, key)
    }

    fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error> {
        RedisJsonCache::set(self, key, value, Some(ttl_seconds)).map(|_stored| ())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CachedJsonLoad {
    pub data: Option<Value>,
    pub diagnostics: Vec<String>,
    /// True only when this call fetched and persisted a new provider response.
    pub refreshed: bool,
}

#[derive(Debug, Deserialize)]
struct CachedResponse {
    timestamp: i64,
    data: Value,
}

#[must_use]
pub fn python_json_string(value: &str) -> String {
    let mut encoded = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => encoded.push_str("\\\""),
            '\\' => encoded.push_str("\\\\"),
            '\u{08}' => encoded.push_str("\\b"),
            '\u{0c}' => encoded.push_str("\\f"),
            '\n' => encoded.push_str("\\n"),
            '\r' => encoded.push_str("\\r"),
            '\t' => encoded.push_str("\\t"),
            character if character <= '\u{1f}' => {
                encoded.push_str(&format!("\\u{:04x}", u32::from(character)));
            }
            character if character.is_ascii() => encoded.push(character),
            character => {
                let codepoint = u32::from(character);
                if codepoint <= 0xffff {
                    encoded.push_str(&format!("\\u{codepoint:04x}"));
                } else {
                    let adjusted = codepoint - 0x1_0000;
                    let high = 0xd800 + (adjusted >> 10);
                    let low = 0xdc00 + (adjusted & 0x3ff);
                    encoded.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                }
            }
        }
    }
    encoded.push('"');
    encoded
}

#[must_use]
pub fn python_request_cache_key(arguments: &str) -> String {
    let hash = Sha256::digest(arguments.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    bot_core::cache_policy::request_cache_key(&hash)
}

pub fn load_cached_json<C, Fetch, Retry>(
    cache: &mut C,
    key: &str,
    ttl_seconds: i64,
    now_unix: i64,
    request_label: &str,
    mut fetch: Fetch,
    mut before_retry: Retry,
) -> CachedJsonLoad
where
    C: RequestCache,
    Fetch: FnMut() -> Result<JsonHttpResponse, String>,
    Retry: FnMut(),
{
    let mut diagnostics = Vec::new();
    let cached = match cache.get(key) {
        Ok(Some(raw)) => match serde_json::from_str::<CachedResponse>(&raw) {
            Ok(cached) => Some(cached),
            Err(error) => {
                diagnostics.push(format!("invalid request cache key {key}: {error}"));
                return CachedJsonLoad {
                    data: None,
                    diagnostics,
                    refreshed: false,
                };
            }
        },
        Ok(None) => None,
        Err(error) => {
            diagnostics.push(format!("could not read request cache key {key}: {error}"));
            return CachedJsonLoad {
                data: None,
                diagnostics,
                refreshed: false,
            };
        }
    };
    if let Some(cached) = &cached
        && now_unix.saturating_sub(cached.timestamp) <= ttl_seconds
    {
        return CachedJsonLoad {
            data: Some(cached.data.clone()),
            diagnostics,
            refreshed: false,
        };
    }

    for attempt in 0..2 {
        let fetched = match fetch() {
            Ok(response) if response.status_code < 400 => {
                serde_json::from_str::<Value>(&response.body)
                    .map_err(|error| format!("invalid JSON: {error}"))
            }
            Ok(response) => Err(format!("HTTP {}", response.status_code)),
            Err(error) => Err(error),
        };
        match fetched {
            Ok(data) => {
                let value = json!({"timestamp": now_unix, "data": data});
                let encoded = match serde_json::to_string(&value) {
                    Ok(encoded) => encoded,
                    Err(error) => {
                        diagnostics.push(format!(
                            "could not encode {request_label} cache value: {error}"
                        ));
                        return CachedJsonLoad {
                            data: cached.map(|cached| cached.data),
                            diagnostics,
                            refreshed: false,
                        };
                    }
                };
                match cache.set(key, &encoded, request_cache_ttl(ttl_seconds)) {
                    Ok(()) => {
                        return CachedJsonLoad {
                            data: value.get("data").cloned(),
                            diagnostics,
                            refreshed: true,
                        };
                    }
                    Err(error) => diagnostics.push(format!(
                        "could not write request cache key {key} on attempt {}: {error}",
                        attempt + 1
                    )),
                }
            }
            Err(error) => diagnostics.push(format!(
                "{request_label} attempt {} failed: {error}",
                attempt + 1
            )),
        }
        if attempt == 0 {
            before_retry();
        }
    }
    CachedJsonLoad {
        data: cached.map(|cached| cached.data),
        diagnostics,
        refreshed: false,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::{
        JsonHttpResponse, RequestCache, load_cached_json, python_json_string,
        python_request_cache_key,
    };

    #[derive(Default)]
    struct Cache {
        gets: VecDeque<Result<Option<String>, &'static str>>,
        sets: VecDeque<Result<(), &'static str>>,
        writes: Vec<(String, String, i64)>,
    }

    impl RequestCache for Cache {
        type Error = &'static str;

        fn get(&mut self, _key: &str) -> Result<Option<String>, Self::Error> {
            self.gets.pop_front().unwrap_or(Ok(None))
        }

        fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error> {
            self.writes
                .push((key.to_owned(), value.to_owned(), ttl_seconds));
            self.sets.pop_front().unwrap_or(Ok(()))
        }
    }

    #[test]
    fn fresh_cache_is_authoritative_and_future_timestamps_are_fresh() {
        for timestamp in [100, 200] {
            let mut cache = Cache {
                gets: VecDeque::from([Ok(Some(format!(
                    r#"{{"timestamp":{timestamp},"data":{{"value":1}}}}"#
                )))]),
                ..Cache::default()
            };
            let mut fetches = 0;
            let load = load_cached_json(
                &mut cache,
                "request_cache:key",
                60,
                100,
                "synthetic request",
                || {
                    fetches += 1;
                    Err("unexpected".to_owned())
                },
                || {},
            );
            assert_eq!(load.data, Some(serde_json::json!({"value": 1})));
            assert_eq!(fetches, 0);
            assert!(load.diagnostics.is_empty());
        }
    }

    #[test]
    fn missing_cache_fetches_writes_compatible_envelope_and_minimum_ttl() {
        let mut cache = Cache::default();
        let load = load_cached_json(
            &mut cache,
            "request_cache:key",
            10,
            123,
            "synthetic request",
            || {
                Ok(JsonHttpResponse {
                    status_code: 200,
                    body: r#"{"value":2}"#.to_owned(),
                })
            },
            || {},
        );
        assert_eq!(load.data, Some(serde_json::json!({"value": 2})));
        assert_eq!(cache.writes[0].0, "request_cache:key");
        assert_eq!(cache.writes[0].2, 60);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&cache.writes[0].1).ok(),
            Some(serde_json::json!({"timestamp":123,"data":{"value":2}}))
        );
    }

    #[test]
    fn stale_cache_survives_two_typed_fetch_failures() {
        let mut cache = Cache {
            gets: VecDeque::from([Ok(Some(
                r#"{"timestamp":1,"data":{"stale":true}}"#.to_owned(),
            ))]),
            ..Cache::default()
        };
        let failures = VecDeque::from([
            Ok(JsonHttpResponse {
                status_code: 503,
                body: String::new(),
            }),
            Err("timeout".to_owned()),
        ]);
        let mut failures = failures;
        let mut retries = 0;
        let load = load_cached_json(
            &mut cache,
            "request_cache:key",
            60,
            1_000,
            "synthetic request",
            || failures.pop_front().unwrap_or(Err("missing".to_owned())),
            || retries += 1,
        );
        assert_eq!(load.data, Some(serde_json::json!({"stale": true})));
        assert_eq!(retries, 1);
        assert!(load.diagnostics[0].contains("HTTP 503"));
        assert!(load.diagnostics[1].contains("timeout"));
    }

    #[test]
    fn invalid_json_and_failed_cache_writes_retry_then_fail_closed() {
        let mut cache = Cache {
            sets: VecDeque::from([Err("write one"), Err("write two")]),
            ..Cache::default()
        };
        let responses = VecDeque::from([
            Ok(JsonHttpResponse {
                status_code: 200,
                body: "not-json".to_owned(),
            }),
            Ok(JsonHttpResponse {
                status_code: 200,
                body: r#"{"ok":true}"#.to_owned(),
            }),
        ]);
        let mut responses = responses;
        let load = load_cached_json(
            &mut cache,
            "request_cache:key",
            300,
            100,
            "synthetic request",
            || responses.pop_front().unwrap_or(Err("missing".to_owned())),
            || {},
        );
        assert!(load.data.is_none());
        assert_eq!(load.diagnostics.len(), 2);

        let mut cache = Cache {
            gets: VecDeque::from([Err("read failed")]),
            ..Cache::default()
        };
        let load = load_cached_json(
            &mut cache,
            "request_cache:key",
            300,
            100,
            "synthetic request",
            || Err("must not fetch".to_owned()),
            || {},
        );
        assert!(load.data.is_none());
        assert!(load.diagnostics[0].contains("read failed"));
    }

    #[test]
    fn malformed_cached_envelope_fails_closed_without_fetching() {
        let mut cache = Cache {
            gets: VecDeque::from([Ok(Some(r#"{"timestamp":"bad"}"#.to_owned()))]),
            ..Cache::default()
        };
        let mut fetched = false;
        let load = load_cached_json(
            &mut cache,
            "request_cache:key",
            300,
            100,
            "synthetic request",
            || {
                fetched = true;
                Err("must not fetch".to_owned())
            },
            || {},
        );
        assert!(load.data.is_none());
        assert!(!fetched);
        assert!(load.diagnostics[0].contains("invalid request cache"));
    }

    #[test]
    fn python_string_and_hash_helpers_preserve_ascii_unicode_and_surrogates() {
        assert_eq!(
            python_json_string("quote \" slash \\ newline\n Córdoba 😀"),
            r#""quote \" slash \\ newline\n C\u00f3rdoba \ud83d\ude00""#
        );
        assert_eq!(
            python_request_cache_key("synthetic"),
            "request_cache:b3cc0475bb78a5026098858e9889acf666d31062d513d303314eca31d36e72f2"
        );
    }
}
