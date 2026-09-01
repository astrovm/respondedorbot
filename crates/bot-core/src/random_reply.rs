//! Deterministic outcome mapping for spontaneous yes/no replies.

/// The localized answer key selected by the random source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RandomAnswer {
    Yes,
    No,
}

/// An optional suffix appended to the answer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RandomSuffix {
    None,
    Address,
    Name,
}

/// A deterministic reply outcome produced from adapter-owned random samples.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RandomReply {
    pub answer: RandomAnswer,
    pub suffix: RandomSuffix,
}

/// Map the two legacy random samples to a typed response.
#[must_use]
pub const fn evaluate_random_reply(response_sample: i64, suffix_sample: i64) -> RandomReply {
    RandomReply {
        answer: if response_sample == 0 {
            RandomAnswer::No
        } else {
            RandomAnswer::Yes
        },
        suffix: match suffix_sample {
            1 => RandomSuffix::Address,
            2 => RandomSuffix::Name,
            _ => RandomSuffix::None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{RandomAnswer, RandomReply, RandomSuffix, evaluate_random_reply};

    #[test]
    fn maps_every_legacy_sample_pair() {
        let cases = [
            (0, 0, RandomAnswer::No, RandomSuffix::None),
            (0, 1, RandomAnswer::No, RandomSuffix::Address),
            (0, 2, RandomAnswer::No, RandomSuffix::Name),
            (1, 0, RandomAnswer::Yes, RandomSuffix::None),
            (1, 1, RandomAnswer::Yes, RandomSuffix::Address),
            (1, 2, RandomAnswer::Yes, RandomSuffix::Name),
        ];
        for (response, suffix, answer, expected_suffix) in cases {
            assert_eq!(
                evaluate_random_reply(response, suffix),
                RandomReply {
                    answer,
                    suffix: expected_suffix
                }
            );
        }
    }

    #[test]
    fn preserves_python_truthiness_and_unknown_suffix_behavior() {
        assert_eq!(
            evaluate_random_reply(-4, 99),
            RandomReply {
                answer: RandomAnswer::Yes,
                suffix: RandomSuffix::None,
            }
        );
    }
}
