//! Pure Telegram link-entity parsing helpers.

use std::collections::HashSet;

/// Slice text by Telegram's UTF-16 code-unit offsets, dropping incomplete
/// surrogate pairs in the same way as the legacy adapter's decoding policy.
#[must_use]
pub fn utf16_slice(text: &str, offset: i64, length: i64) -> String {
    if text.is_empty() || length <= 0 {
        return String::new();
    }
    let units = text.encode_utf16().collect::<Vec<_>>();
    let start = usize::try_from(offset.max(0))
        .unwrap_or(usize::MAX)
        .min(units.len());
    let requested_length = usize::try_from(length.max(0)).unwrap_or(usize::MAX);
    let end = start.saturating_add(requested_length).min(units.len());
    char::decode_utf16(units[start..end].iter().copied())
        .filter_map(Result::ok)
        .collect()
}

/// Remove message punctuation that Telegram's broad URL matcher may capture.
#[must_use]
pub fn trim_detected_url(raw_url: &str) -> String {
    raw_url
        .trim()
        .trim_end_matches(&['.', ',', ';', ':', '!', '?', ')', '"', ']', '}', '\''][..])
        .to_owned()
}

/// Keep the first occurrence of each URL and apply the configured message
/// limit. The legacy implementation returns one URL for a zero limit.
#[must_use]
pub fn select_unique_urls(candidates: &[String], max_links: usize) -> Vec<String> {
    let effective_limit = max_links.max(1);
    let mut seen = HashSet::new();
    let mut selected = Vec::new();
    for candidate in candidates {
        if seen.insert(candidate.as_str()) {
            selected.push(candidate.clone());
            if selected.len() >= effective_limit {
                break;
            }
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::{select_unique_urls, trim_detected_url, utf16_slice};

    #[test]
    fn slices_telegram_utf16_offsets_across_emoji() {
        let text = "a😀 link";
        assert_eq!(utf16_slice(text, 3, 5), " link");
        assert_eq!(utf16_slice(text, 1, 1), "");
        assert_eq!(utf16_slice(text, -5, 1), "a");
        assert_eq!(utf16_slice(text, 0, 0), "");
        assert_eq!(utf16_slice("", 0, 2), "");
    }

    #[test]
    fn trims_only_the_legacy_url_suffix_characters() {
        assert_eq!(
            trim_detected_url("  https://example.test/path).  "),
            "https://example.test/path"
        );
        assert_eq!(
            trim_detected_url("https://example.test/path("),
            "https://example.test/path("
        );
    }

    #[test]
    fn deduplicates_stably_and_preserves_the_zero_limit_quirk() {
        let candidates = vec![
            "https://a.test".to_owned(),
            "https://a.test".to_owned(),
            "https://b.test".to_owned(),
            "https://c.test".to_owned(),
        ];
        assert_eq!(
            select_unique_urls(&candidates, 2),
            vec!["https://a.test".to_owned(), "https://b.test".to_owned()]
        );
        assert_eq!(
            select_unique_urls(&candidates, 0),
            vec!["https://a.test".to_owned()]
        );
        assert!(select_unique_urls(&[], 3).is_empty());
    }
}
