use chrono::{DateTime, Utc};
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    pub value: Option<T>,
    pub expires_at: SystemTime,
    pub stale_until: SystemTime,
    pub fetched_at: DateTime<Utc>,
}

impl<T> CacheEntry<T> {
    pub fn empty() -> Self {
        Self {
            value: None,
            expires_at: SystemTime::UNIX_EPOCH,
            stale_until: SystemTime::UNIX_EPOCH,
            fetched_at: DateTime::<Utc>::from(SystemTime::UNIX_EPOCH),
        }
    }
}

pub fn update_local_cache<T>(entry: &mut CacheEntry<T>, value: T, ttl: Duration, stale_grace: Duration) {
    let now = SystemTime::now();
    entry.value = Some(value);
    entry.expires_at = now + ttl;
    entry.stale_until = entry.expires_at + stale_grace;
    entry.fetched_at = DateTime::<Utc>::from(now);
}

pub fn local_cache_get<T: Clone>(entry: &mut CacheEntry<T>, allow_stale: bool) -> (Option<T>, bool, DateTime<Utc>) {
    let now = SystemTime::now();
    let Some(value) = entry.value.clone() else {
        return (None, false, entry.fetched_at);
    };

    if now <= entry.expires_at {
        return (Some(value), true, entry.fetched_at);
    }

    if allow_stale && now <= entry.stale_until {
        return (Some(value), false, entry.fetched_at);
    }

    entry.value = None;
    (None, false, entry.fetched_at)
}
