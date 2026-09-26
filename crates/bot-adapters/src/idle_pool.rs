//! Age-limited idle list shared by the PostgreSQL and Redis pools.
//!
//! A server or pooler (for example Supabase's) may close a connection while it
//! sits idle, and a synchronous client only notices on the next query. Idle
//! connections older than [`MAX_IDLE_AGE`] are therefore discarded instead of
//! reused, so the first query after a quiet period opens a fresh connection.

use std::time::{Duration, Instant};

/// How long a returned connection may stay idle before it is discarded.
pub(crate) const MAX_IDLE_AGE: Duration = Duration::from_secs(60);

/// Connections returned to a pool, each with the time it was returned.
pub(crate) struct IdleList<T> {
    entries: Vec<(Instant, T)>,
}

impl<T> IdleList<T> {
    pub(crate) const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Store a connection returned at `returned_at`.
    pub(crate) fn push(&mut self, returned_at: Instant, value: T) {
        self.entries.push((returned_at, value));
    }

    /// Drop every connection idle for longer than `max_age` at `now`, then
    /// return the most recently returned remaining one.
    pub(crate) fn pop_fresh(&mut self, now: Instant, max_age: Duration) -> Option<T> {
        self.entries
            .retain(|(returned_at, _)| now.saturating_duration_since(*returned_at) <= max_age);
        self.entries.pop().map(|(_, value)| value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pop_fresh_returns_most_recent_connection_within_age_limit() {
        let start = Instant::now();
        let mut idle = IdleList::new();
        idle.push(start, "first");
        idle.push(start + Duration::from_secs(1), "second");
        assert_eq!(idle.len(), 2);

        let now = start + Duration::from_secs(30);
        assert_eq!(idle.pop_fresh(now, MAX_IDLE_AGE), Some("second"));
        assert_eq!(idle.pop_fresh(now, MAX_IDLE_AGE), Some("first"));
        assert_eq!(idle.pop_fresh(now, MAX_IDLE_AGE), None);
    }

    #[test]
    fn pop_fresh_discards_connections_idle_past_the_limit() {
        let start = Instant::now();
        let mut idle = IdleList::new();
        idle.push(start, "stale");
        idle.push(start + Duration::from_secs(50), "fresh");

        let now = start + Duration::from_secs(61);
        assert_eq!(idle.pop_fresh(now, MAX_IDLE_AGE), Some("fresh"));
        assert_eq!(idle.len(), 0);

        idle.push(start, "stale again");
        assert_eq!(idle.pop_fresh(now, MAX_IDLE_AGE), None);
        assert_eq!(idle.len(), 0);
    }
}
