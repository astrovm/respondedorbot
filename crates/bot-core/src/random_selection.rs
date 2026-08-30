//! Parsing for the `/random` command, with randomness supplied by an adapter.

use num_bigint::BigInt;

/// A validated random operation for an outer adapter to execute.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RandomSelection {
    Choices { values: Vec<String> },
    InclusiveRange { start: BigInt, end: BigInt },
    Invalid,
}

/// Boundary incompatibility that should use the legacy implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsupportedUnicodeRange;

/// Parse one legacy random-selection request without consuming randomness.
pub fn parse_random_selection(input: &str) -> Result<RandomSelection, UnsupportedUnicodeRange> {
    let values: Vec<_> = input
        .split(',')
        .map(|value| value.trim().to_owned())
        .collect();
    if values.len() >= 2 {
        return Ok(RandomSelection::Choices { values });
    }

    if !input.is_ascii() {
        return Err(UnsupportedUnicodeRange);
    }
    let range: Vec<_> = input.split('-').map(str::trim).collect();
    if range.len() != 2 {
        return Ok(RandomSelection::Invalid);
    }
    let (Some(start), Some(end)) = (
        BigInt::parse_bytes(range[0].as_bytes(), 10),
        BigInt::parse_bytes(range[1].as_bytes(), 10),
    ) else {
        return Ok(RandomSelection::Invalid);
    };
    if start >= end {
        return Ok(RandomSelection::Invalid);
    }
    Ok(RandomSelection::InclusiveRange { start, end })
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;

    use super::{RandomSelection, UnsupportedUnicodeRange, parse_random_selection};

    #[test]
    fn parses_comma_choices_with_legacy_whitespace_and_empty_values() {
        assert_eq!(
            parse_random_selection(" pizza, pasta , "),
            Ok(RandomSelection::Choices {
                values: vec!["pizza".to_owned(), "pasta".to_owned(), String::new()]
            })
        );
        assert_eq!(
            parse_random_selection("uno,🧉"),
            Ok(RandomSelection::Choices {
                values: vec!["uno".to_owned(), "🧉".to_owned()]
            })
        );
    }

    #[test]
    fn parses_arbitrary_precision_inclusive_ranges() {
        assert_eq!(
            parse_random_selection("100000000000000000000-100000000000000000002"),
            Ok(RandomSelection::InclusiveRange {
                start: BigInt::from(100_u8) * BigInt::from(10_u8).pow(18),
                end: BigInt::from(100_u8) * BigInt::from(10_u8).pow(18) + BigInt::from(2_u8),
            })
        );
    }

    #[test]
    fn rejects_malformed_descending_and_negative_ranges() {
        for input in ["invalid", "1", "3-3", "4-3", "-2-3", "1-2-3"] {
            assert_eq!(parse_random_selection(input), Ok(RandomSelection::Invalid));
        }
    }

    #[test]
    fn asks_the_adapter_to_preserve_unicode_integer_semantics() {
        assert_eq!(
            parse_random_selection("１-３"),
            Err(UnsupportedUnicodeRange)
        );
    }
}
