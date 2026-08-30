//! Deterministic Polymarket outcome reconciliation and ranking.

use std::cmp::Ordering;

#[derive(Clone, Debug, PartialEq)]
pub struct MarketOutcome {
    pub title: String,
    pub cached_probability: f64,
    pub live_probability: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RankedOutcome {
    pub title: String,
    pub percentage: f64,
}

/// Reconcile optional live prices, clamp probabilities, and return the highest
/// outcomes. Sorting is stable so provider order breaks equal-price ties.
#[must_use]
pub fn rank_outcomes(outcomes: &[MarketOutcome], limit: usize) -> Vec<RankedOutcome> {
    let mut ranked = outcomes
        .iter()
        .map(|outcome| RankedOutcome {
            title: outcome.title.clone(),
            percentage: outcome
                .live_probability
                .unwrap_or(outcome.cached_probability)
                .clamp(0.0, 1.0)
                * 100.0,
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .percentage
            .partial_cmp(&left.percentage)
            .unwrap_or(Ordering::Equal)
    });
    ranked.truncate(limit);
    ranked
}

#[cfg(test)]
mod tests {
    use super::{MarketOutcome, RankedOutcome, rank_outcomes};

    fn outcome(title: &str, cached: f64, live: Option<f64>) -> MarketOutcome {
        MarketOutcome {
            title: title.to_owned(),
            cached_probability: cached,
            live_probability: live,
        }
    }

    #[test]
    fn live_prices_override_cached_prices_before_ranking() {
        let ranked = rank_outcomes(
            &[
                outcome("cached leader", 0.9, Some(0.2)),
                outcome("live leader", 0.4, Some(0.8)),
                outcome("cached only", 0.6, None),
            ],
            2,
        );
        assert_eq!(
            ranked,
            vec![
                RankedOutcome {
                    title: "live leader".to_owned(),
                    percentage: 80.0,
                },
                RankedOutcome {
                    title: "cached only".to_owned(),
                    percentage: 60.0,
                },
            ]
        );
    }

    #[test]
    fn probabilities_are_clamped_and_equal_values_keep_provider_order() {
        assert_eq!(
            rank_outcomes(
                &[
                    outcome("first", 2.0, None),
                    outcome("second", 1.0, Some(3.0)),
                    outcome("last", -1.0, None),
                ],
                10,
            ),
            vec![
                RankedOutcome {
                    title: "first".to_owned(),
                    percentage: 100.0,
                },
                RankedOutcome {
                    title: "second".to_owned(),
                    percentage: 100.0,
                },
                RankedOutcome {
                    title: "last".to_owned(),
                    percentage: 0.0,
                },
            ]
        );
    }

    #[test]
    fn zero_limit_and_empty_inputs_are_supported() {
        assert!(rank_outcomes(&[outcome("ignored", 0.5, None)], 0).is_empty());
        assert!(rank_outcomes(&[], 3).is_empty());
    }
}
