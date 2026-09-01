//! Durable memory-compaction job state transitions.

use serde::Serialize;

pub const MAX_ATTEMPTS: u32 = 3;
const RETRY_BASE_SECONDS: f64 = 30.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionDisposition {
    SettleRecoveredSuccess,
    SettleObsolete,
    GenerateSummary,
    SaveAndSettle,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct RetryTransition {
    pub attempts: u32,
    pub terminal: bool,
    pub next_attempt_at: Option<f64>,
    pub actual_credit_units: Option<i64>,
}

/// Check whether a durable job may attempt execution at the current clock value.
#[must_use]
pub fn is_due(next_attempt_at: f64, now: f64) -> bool {
    next_attempt_at <= now
}

/// Decide the next action after any durable provider result has been restored.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn evaluate_compaction(
    current_summary: Option<&str>,
    current_marker: Option<&str>,
    prior_summary: Option<&str>,
    expected_marker: Option<&str>,
    result_summary: Option<&str>,
    target_marker: &str,
) -> CompactionDisposition {
    if result_summary.is_some_and(|result| {
        current_summary == Some(result) && current_marker == Some(target_marker)
    }) {
        return CompactionDisposition::SettleRecoveredSuccess;
    }
    if current_summary != prior_summary || current_marker != expected_marker {
        return CompactionDisposition::SettleObsolete;
    }
    if result_summary.is_none_or(str::is_empty) {
        CompactionDisposition::GenerateSummary
    } else {
        CompactionDisposition::SaveAndSettle
    }
}

/// Advance the bounded exponential-retry policy after one failed attempt.
#[must_use]
pub fn retry_after_failure(attempts: u32, now: f64, has_billing_segment: bool) -> RetryTransition {
    let attempts = attempts.saturating_add(1);
    if attempts >= MAX_ATTEMPTS {
        return RetryTransition {
            attempts,
            terminal: true,
            next_attempt_at: None,
            actual_credit_units: (!has_billing_segment).then_some(0),
        };
    }
    let exponent = attempts.saturating_sub(1).min(31);
    let delay = RETRY_BASE_SECONDS * f64::from(2_u32.pow(exponent));
    RetryTransition {
        attempts,
        terminal: false,
        next_attempt_at: Some(now + delay),
        actual_credit_units: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CompactionDisposition, MAX_ATTEMPTS, evaluate_compaction, is_due, retry_after_failure,
    };

    #[test]
    fn evaluates_recovery_obsolescence_generation_and_save_paths() {
        assert_eq!(
            evaluate_compaction(
                Some("result"),
                Some("m2"),
                Some("old"),
                Some("m1"),
                Some("result"),
                "m2",
            ),
            CompactionDisposition::SettleRecoveredSuccess
        );
        assert_eq!(
            evaluate_compaction(
                Some("new"),
                Some("m3"),
                Some("old"),
                Some("m1"),
                Some("result"),
                "m2",
            ),
            CompactionDisposition::SettleObsolete
        );
        assert_eq!(
            evaluate_compaction(None, None, None, None, None, "m2"),
            CompactionDisposition::GenerateSummary
        );
        assert_eq!(
            evaluate_compaction(None, None, None, None, Some("result"), "m2"),
            CompactionDisposition::SaveAndSettle
        );
    }

    #[test]
    fn applies_due_boundary_and_bounded_exponential_retry() {
        assert!(!is_due(101.0, 100.0));
        assert!(is_due(100.0, 100.0));
        let first = retry_after_failure(0, 100.0, false);
        assert_eq!(first.attempts, 1);
        assert_eq!(first.next_attempt_at, Some(130.0));
        let second = retry_after_failure(1, 100.0, false);
        assert_eq!(second.next_attempt_at, Some(160.0));
        let terminal_without_usage = retry_after_failure(MAX_ATTEMPTS - 1, 100.0, false);
        assert!(terminal_without_usage.terminal);
        assert_eq!(terminal_without_usage.actual_credit_units, Some(0));
        let terminal_with_usage = retry_after_failure(MAX_ATTEMPTS - 1, 100.0, true);
        assert_eq!(terminal_with_usage.actual_credit_units, None);
    }
}
