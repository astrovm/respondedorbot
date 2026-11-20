//! Which asset people mean by an ambiguous price query, learned from the
//! asset each person last kept or switched to.
//!
//! `/p ANT` matches several assets. The bot shows the most likely one and
//! lets the person switch. Each person has one vote per query: the asset they
//! last switched to, or the one they were shown if they never switched. So
//! people who got the right answer and moved on count too, and a few switches
//! cannot override everyone who kept the first answer.

use serde::{Deserialize, Serialize};

use crate::market_prices::MarketCandidate;

/// How many people's votes one query keeps; the oldest votes go first.
pub const MAX_VOTES: usize = 500;

/// How long a query's votes are kept after the last vote.
pub const VOTES_TTL_SECONDS: i64 = 180 * 86_400;

/// The cache key of a query's votes. `scope` keeps `/s`, `/c` and `/p`
/// apart, because each command searches a different set of assets.
#[must_use]
pub fn choices_key(scope: &str, query: &str) -> String {
    let query = query.trim().trim_start_matches('$').to_lowercase();
    format!("market_choices:{scope}:{query}")
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarketChoices {
    /// `(user id, candidate id)`, oldest vote first.
    #[serde(default)]
    pub votes: Vec<(i64, String)>,
}

impl MarketChoices {
    /// Records a person's choice. A kept answer (`switched` false) never
    /// replaces an earlier switch: that person was shown the asset they
    /// had chosen, or it was missing from this search.
    pub fn record(&mut self, user_id: i64, candidate_id: &str, switched: bool) {
        let existing = self.votes.iter().position(|(user, _)| *user == user_id);
        if let Some(index) = existing {
            if !switched {
                return;
            }
            self.votes.remove(index);
        }
        self.votes.push((user_id, candidate_id.to_owned()));
        let excess = self.votes.len().saturating_sub(MAX_VOTES);
        self.votes.drain(..excess);
    }

    fn own(&self, user_id: i64) -> Option<&str> {
        self.votes
            .iter()
            .find(|(user, _)| *user == user_id)
            .map(|(_, candidate)| candidate.as_str())
    }

    /// The asset with the most votes among `candidates`, if one leads.
    /// With no votes, a lone candidate leads, which leaves the order as is.
    fn leader<'a>(&self, candidates: &'a [MarketCandidate]) -> Option<&'a str> {
        let counts = candidates
            .iter()
            .map(|candidate| {
                let votes = self
                    .votes
                    .iter()
                    .filter(|(_, chosen)| *chosen == candidate.id)
                    .count();
                (votes, candidate.id.as_str())
            })
            .collect::<Vec<_>>();
        let most = counts.iter().map(|(votes, _)| *votes).max()?;
        let mut leaders = counts.iter().filter(|(votes, _)| *votes == most);
        match (leaders.next(), leaders.next()) {
            (Some((_, id)), None) => Some(id),
            _ => None,
        }
    }

    /// Moves the asset this person last chose to the front, or else the one
    /// most people chose. Otherwise the ranked order stays.
    pub fn order(&self, candidates: &mut [MarketCandidate], user_id: Option<i64>) {
        let own = user_id
            .and_then(|user_id| self.own(user_id))
            .filter(|own| candidates.iter().any(|candidate| candidate.id == *own))
            .map(str::to_owned);
        let preferred = own.or_else(|| self.leader(candidates).map(str::to_owned));
        if let Some(index) = preferred.and_then(|preferred| {
            candidates
                .iter()
                .position(|candidate| candidate.id == preferred)
        }) {
            candidates[..=index].rotate_right(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_VOTES, MarketChoices, VOTES_TTL_SECONDS, choices_key};
    use crate::market_prices::MarketCandidate;

    fn candidates(ids: &[&str]) -> Vec<MarketCandidate> {
        ids.iter()
            .map(|id| MarketCandidate {
                id: (*id).to_owned(),
                symbol: (*id).to_owned(),
                name: (*id).to_owned(),
                slug: id.to_lowercase(),
                price: "1".to_owned(),
                change: "0".to_owned(),
                currency: String::new(),
                exchange: String::new(),
                asset_type: String::new(),
                contracts: Vec::new(),
            })
            .collect()
    }

    fn ordered(choices: &MarketChoices, user_id: Option<i64>) -> Vec<String> {
        let mut list = candidates(&["stock:ANT", "ANT", "stock:ANT.MX"]);
        choices.order(&mut list, user_id);
        list.into_iter().map(|candidate| candidate.id).collect()
    }

    #[test]
    fn keys_ignore_case_spacing_and_dollar_signs() {
        assert_eq!(choices_key("any", " $Ant "), "market_choices:any:ant");
        assert_eq!(choices_key("stock", "ANT"), "market_choices:stock:ant");
    }

    #[test]
    fn people_who_kept_the_answer_outvote_a_few_switches() {
        let mut choices = MarketChoices::default();
        for user in 1..=5 {
            choices.record(user, "stock:ANT", false);
        }
        for user in 6..=8 {
            choices.record(user, "ANT", false);
            choices.record(user, "ANT", true);
        }
        assert_eq!(
            ordered(&choices, None),
            ["stock:ANT", "ANT", "stock:ANT.MX"]
        );
        // More switches than keeps move the coin first, keeping the rest.
        for user in 1..=2 {
            choices.record(user, "ANT", true);
        }
        assert_eq!(
            ordered(&choices, None),
            ["ANT", "stock:ANT", "stock:ANT.MX"]
        );
        // A tie keeps the ranked order.
        choices.record(3, "ANT", false);
        choices.record(9, "stock:ANT", false);
        choices.record(10, "stock:ANT", false);
        assert_eq!(
            ordered(&choices, None),
            ["stock:ANT", "ANT", "stock:ANT.MX"]
        );
    }

    #[test]
    fn a_person_gets_their_own_last_choice_first() {
        let mut choices = MarketChoices::default();
        for user in 1..=5 {
            choices.record(user, "stock:ANT", false);
        }
        choices.record(9, "stock:ANT", false);
        choices.record(9, "stock:ANT.MX", true);
        // A later kept answer does not undo the switch.
        choices.record(9, "stock:ANT", false);
        assert_eq!(
            ordered(&choices, Some(9)),
            ["stock:ANT.MX", "stock:ANT", "ANT"]
        );
        assert_eq!(
            ordered(&choices, Some(1)),
            ["stock:ANT", "ANT", "stock:ANT.MX"]
        );
        // An own choice missing from this search falls back to the leader.
        let mut missing = MarketChoices::default();
        missing.record(1, "ANT", false);
        missing.record(9, "stock:ANTX", true);
        assert_eq!(
            ordered(&missing, Some(9)),
            ["ANT", "stock:ANT", "stock:ANT.MX"]
        );
        // With no votes for these assets, the ranked order stays.
        assert_eq!(
            ordered(&MarketChoices::default(), Some(9)),
            ["stock:ANT", "ANT", "stock:ANT.MX"]
        );
        let mut lone = MarketChoices::default();
        lone.record(1, "stock:ANT.MX", false);
        assert_eq!(ordered(&lone, None), ["stock:ANT.MX", "stock:ANT", "ANT"]);
    }

    #[test]
    fn only_the_latest_votes_are_kept() {
        let mut choices = MarketChoices::default();
        for user in 0..(MAX_VOTES as i64 + 3) {
            choices.record(user, "ANT", false);
        }
        assert_eq!(choices.votes.len(), MAX_VOTES);
        assert_eq!(choices.votes[0].0, 3);
        // Votes are kept for about six months.
        assert_eq!(VOTES_TTL_SECONDS, 15_552_000);
        let decoded: MarketChoices = serde_json::from_str(r#"{}"#).unwrap_or_default();
        assert!(decoded.votes.is_empty());
    }
}
