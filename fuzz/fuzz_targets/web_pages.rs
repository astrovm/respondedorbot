//! Fetched web pages and URLs the AI asks to open.

#![no_main]

use bot_adapters::web_fetch::{extract_text_from_html, normalize_http_url};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|text: &str| {
    let _ = extract_text_from_html(text);
    if let Some(url) = normalize_http_url(text) {
        assert!(matches!(url.scheme(), "http" | "https"));
    }
});
