//! Link repair over arbitrary message text and Telegram entity offsets.

#![no_main]

use bot_core::links::{
    has_replaceable_link, replace_social_links, select_unique_urls, trim_detected_url, utf16_slice,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: (&str, i64, i64)| {
    let (text, offset, length) = input;
    let replacement = replace_social_links(text, 1_700_000_000, |_| true);
    // Only a real replacement counts as a change, and each one keeps its original.
    assert_eq!(replacement.changed, !replacement.original_links.is_empty());
    let unchanged = replace_social_links(text, 1_700_000_000, |_| false);
    assert!(!unchanged.changed);
    assert!(unchanged.original_links.is_empty());
    let _ = has_replaceable_link(text);
    let _ = utf16_slice(text, offset, length);
    let trimmed = trim_detected_url(text);
    assert!(trimmed.len() <= text.len());
    let urls = select_unique_urls(&unchanged.original_links, 3);
    assert!(urls.len() <= 3);
});
