//! Cleanup for provider text that is sent through Telegram without Markdown.

use std::sync::LazyLock;

use regex::Regex;

static CLEANUP_RULES: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    [
        (r"\*\*([^*]+)\*\*", "$1"),
        (r"__([^_]+)__", "$1"),
        (r"\*([^*]+)\*", "$1"),
        (r"_([^_]+)_", "$1"),
        (r"(?m)^#{1,6}\s*", ""),
        (r"(?s)```.*?```", ""),
        (r"`([^`]+)`", "$1"),
        (r"\[([^\]]+)\]\([^)]+\)", "$1"),
        (r"(?m)^---+\s*$", ""),
        (r"\n{3,}", "\n\n"),
    ]
    .into_iter()
    .filter_map(|(pattern, replacement)| Regex::new(pattern).ok().map(|regex| (regex, replacement)))
    .collect()
});

#[must_use]
pub fn sanitize_summary_text(text: &str) -> String {
    CLEANUP_RULES
        .iter()
        .fold(text.to_owned(), |text, (regex, replacement)| {
            regex.replace_all(&text, *replacement).into_owned()
        })
        .trim()
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::sanitize_summary_text;

    #[test]
    fn removes_telegram_unsafe_markdown_with_python_parity() {
        assert_eq!(
            sanitize_summary_text(
                "# title\n\n**bold** _italic_ [source](https://example.test)\n\n```secret```\n\n\n`code`"
            ),
            "title\n\nbold italic source\n\ncode"
        );
    }

    #[test]
    fn trims_plain_text_without_changing_content() {
        assert_eq!(
            sanitize_summary_text("  synthetic text  "),
            "synthetic text"
        );
    }
}
