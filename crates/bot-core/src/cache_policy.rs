//! Redis cache key compatibility and stale-while-refresh decisions.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheDecision {
    Fresh,
    ServeStale,
    RefreshInline,
}

/// Decide how to handle one optional cache timestamp.
#[must_use]
pub fn evaluate_cache(
    cached_timestamp: Option<i64>,
    now: i64,
    ttl: i64,
    stale_grace: i64,
) -> CacheDecision {
    let Some(timestamp) = cached_timestamp else {
        return CacheDecision::RefreshInline;
    };
    let age = now.saturating_sub(timestamp);
    if age <= ttl {
        CacheDecision::Fresh
    } else if age <= ttl.saturating_add(stale_grace) {
        CacheDecision::ServeStale
    } else {
        CacheDecision::RefreshInline
    }
}

#[must_use]
pub fn request_cache_key(request_hash: &str) -> String {
    format!("request_cache:{request_hash}")
}

#[must_use]
pub fn request_cache_history_key(hour_key: &str, request_hash: &str) -> String {
    format!("request_cache_history:{hour_key}:{request_hash}")
}

#[must_use]
pub fn request_cache_ttl(expiration_time: i64) -> i64 {
    expiration_time.max(60)
}

#[must_use]
pub fn last_success_ttl(ttl: i64, stale_grace: i64) -> i64 {
    ttl.saturating_add(stale_grace)
}

#[cfg(test)]
mod tests {
    use super::{
        CacheDecision, evaluate_cache, last_success_ttl, request_cache_history_key,
        request_cache_key, request_cache_ttl,
    };

    #[test]
    fn chooses_fresh_stale_and_inline_refresh_at_exact_boundaries() {
        assert_eq!(
            evaluate_cache(None, 100, 10, 60),
            CacheDecision::RefreshInline
        );
        assert_eq!(evaluate_cache(Some(90), 100, 10, 60), CacheDecision::Fresh);
        assert_eq!(
            evaluate_cache(Some(89), 100, 10, 60),
            CacheDecision::ServeStale
        );
        assert_eq!(
            evaluate_cache(Some(30), 100, 10, 60),
            CacheDecision::ServeStale
        );
        assert_eq!(
            evaluate_cache(Some(29), 100, 10, 60),
            CacheDecision::RefreshInline
        );
        assert_eq!(evaluate_cache(Some(110), 100, 10, 60), CacheDecision::Fresh);
    }

    #[test]
    fn preserves_cache_keys_and_ttl_rules() {
        assert_eq!(request_cache_key("abc"), "request_cache:abc");
        assert_eq!(
            request_cache_history_key("2026-08-30-01", "abc"),
            "request_cache_history:2026-08-30-01:abc"
        );
        assert_eq!(request_cache_ttl(0), 60);
        assert_eq!(request_cache_ttl(120), 120);
        assert_eq!(last_success_ttl(10, 60), 70);
    }
}
