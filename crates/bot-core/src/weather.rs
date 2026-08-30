//! Deterministic selection rules for weather-provider responses.

/// Choose the best geocoding candidate from adapter-normalized search keys.
///
/// Each candidate key contains its administrative region, country, and country
/// code. Qualifier keys come from the requested location. The first candidate
/// wins when no qualifier matches or when several candidates have the same
/// score, matching the existing provider-order behavior.
#[must_use]
pub fn select_location_candidate(
    qualifier_keys: &[String],
    candidate_keys: &[String],
) -> Option<usize> {
    if candidate_keys.is_empty() {
        return None;
    }
    if qualifier_keys.is_empty() {
        return Some(0);
    }

    let mut best_index = 0;
    let mut best_score = 0;
    for (index, candidate) in candidate_keys.iter().enumerate() {
        let score = qualifier_keys
            .iter()
            .filter(|qualifier| candidate.contains(qualifier.as_str()))
            .count();
        if score > best_score {
            best_index = index;
            best_score = score;
        }
    }
    Some(if best_score == 0 { 0 } else { best_index })
}

/// Select the hourly forecast row matching the provider clock or local clock.
///
/// The adapter validates and normalizes each ISO timestamp to an hourly key.
/// Provider time is checked before local time for each row, preserving provider
/// order when either clock identifies an earlier row.
#[must_use]
pub fn select_forecast_hour(
    forecast_hours: &[String],
    provider_hour: Option<&str>,
    local_hour: &str,
) -> Option<usize> {
    forecast_hours
        .iter()
        .position(|hour| provider_hour.is_some_and(|current| hour == current) || hour == local_hour)
}

#[cfg(test)]
mod tests {
    use super::{select_forecast_hour, select_location_candidate};

    #[test]
    fn location_selection_defaults_to_the_first_provider_result() {
        assert_eq!(
            select_location_candidate(&[], &["first".to_owned(), "second".to_owned()]),
            Some(0)
        );
        assert_eq!(
            select_location_candidate(
                &["missing".to_owned()],
                &["first".to_owned(), "second".to_owned()]
            ),
            Some(0)
        );
        assert_eq!(select_location_candidate(&["any".to_owned()], &[]), None);
    }

    #[test]
    fn location_selection_uses_all_matching_qualifiers_and_keeps_ties_stable() {
        let qualifiers = vec!["exampleland".to_owned(), "north".to_owned()];
        let candidates = vec![
            "north otherland no".to_owned(),
            "south exampleland ex".to_owned(),
            "north exampleland ex".to_owned(),
            "north exampleland duplicate".to_owned(),
        ];
        assert_eq!(select_location_candidate(&qualifiers, &candidates), Some(2));
    }

    #[test]
    fn forecast_selection_prefers_the_first_matching_row() {
        let hours = vec![
            "2026-01-02T09".to_owned(),
            "2026-01-02T10".to_owned(),
            "2026-01-02T10".to_owned(),
        ];
        assert_eq!(
            select_forecast_hour(&hours, Some("2026-01-02T10"), "2026-01-02T09"),
            Some(0)
        );
        assert_eq!(
            select_forecast_hour(&hours, Some("2026-01-02T11"), "2026-01-02T10"),
            Some(1)
        );
    }

    #[test]
    fn forecast_selection_reports_missing_rows() {
        assert_eq!(
            select_forecast_hour(&["2026-01-02T09".to_owned()], None, "2026-01-02T10"),
            None
        );
    }
}
