//! Parsing and bounded-output rules for administrative reports.

use num_bigint::BigInt;
use unicode_normalization::UnicodeNormalization;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CreditLogLimit {
    Valid(usize),
    Invalid,
    UnsupportedNumericInput,
}

/// Parse `/creditlog [limit]` and clamp it to the database query bounds.
#[must_use]
pub fn parse_creditlog_limit(message_text: &str) -> CreditLogLimit {
    let raw = message_text.trim();
    if raw.is_empty() {
        return CreditLogLimit::Valid(10);
    }
    let token = raw.split_once(' ').map_or(raw, |(token, _)| token).trim();
    let normalized = token.nfkc().collect::<String>();
    if !normalized.is_ascii() {
        return CreditLogLimit::UnsupportedNumericInput;
    }
    let bytes = normalized.as_bytes();
    let underscores_are_valid = bytes.iter().enumerate().all(|(index, byte)| {
        *byte != b'_'
            || (index > 0
                && index + 1 < bytes.len()
                && bytes[index - 1].is_ascii_digit()
                && bytes[index + 1].is_ascii_digit())
    });
    if !underscores_are_valid {
        return CreditLogLimit::Invalid;
    }
    let compact = normalized.replace('_', "");
    let Some(value) = BigInt::parse_bytes(compact.as_bytes(), 10) else {
        return CreditLogLimit::Invalid;
    };
    let minimum = BigInt::from(1_u8);
    let maximum = BigInt::from(25_u8);
    if value < minimum {
        CreditLogLimit::Valid(1)
    } else if value > maximum {
        CreditLogLimit::Valid(25)
    } else {
        value
            .to_string()
            .parse::<usize>()
            .map_or(CreditLogLimit::Invalid, CreditLogLimit::Valid)
    }
}

/// Truncate a report by Unicode scalar count and append the localized marker.
#[must_use]
pub fn truncate_report(text: &str, max_length: usize, truncated_label: &str) -> String {
    if text.chars().count() <= max_length {
        return text.to_owned();
    }
    let suffix = format!("\n\n[{truncated_label}]");
    let content_length = max_length.saturating_sub(suffix.chars().count());
    let content = text
        .chars()
        .take(content_length)
        .collect::<String>()
        .trim_end()
        .to_owned();
    format!("{content}{suffix}")
}

#[cfg(test)]
mod tests {
    use super::{CreditLogLimit, parse_creditlog_limit, truncate_report};

    #[test]
    fn parses_defaults_clamps_and_rejects_invalid_limits() {
        for (input, expected) in [
            ("", CreditLogLimit::Valid(10)),
            ("  ", CreditLogLimit::Valid(10)),
            ("5", CreditLogLimit::Valid(5)),
            ("5 ignored", CreditLogLimit::Valid(5)),
            ("0", CreditLogLimit::Valid(1)),
            ("999999999999999999999", CreditLogLimit::Valid(25)),
            ("invalid", CreditLogLimit::Invalid),
            ("１", CreditLogLimit::Valid(1)),
            ("1_0", CreditLogLimit::Valid(10)),
            ("1__0", CreditLogLimit::Invalid),
        ] {
            assert_eq!(parse_creditlog_limit(input), expected, "{input}");
        }
    }

    #[test]
    fn truncates_by_unicode_characters_and_keeps_short_reports() {
        assert_eq!(truncate_report("short", 10, "truncated"), "short");
        assert_eq!(
            truncate_report("😀😀😀😀😀 trailing", 10, "x"),
            "😀😀😀😀😀\n\n[x]"
        );
        assert_eq!(truncate_report("abcdef", 3, "long"), "\n\n[long]");
    }
}
