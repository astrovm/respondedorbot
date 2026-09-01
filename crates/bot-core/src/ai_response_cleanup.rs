//! Deterministic cleanup for provider-generated assistant responses.

use regex::Regex;
use std::sync::LazyLock;

macro_rules! static_regex {
    ($name:ident, $pattern:literal) => {
        static $name: LazyLock<Option<Regex>> = LazyLock::new(|| Regex::new($pattern).ok());
    };
}

static_regex!(GORDO_PREFIX, r"(?i)^\s*gordo\b\s*:\s*");
static_regex!(MARKDOWN_CODE_FENCE, r"(?s)```(?:[^\n`]*)\n?(.*?)```");
static_regex!(MARKDOWN_IMAGE, r"!\[([^\]]*)\]\([^)]*\)");
static_regex!(MARKDOWN_LINK, r"\[([^\]]+)\]\([^)]*\)");
static_regex!(MARKDOWN_INLINE_CODE, r"`([^`]+)`");
static_regex!(MARKDOWN_HEADER, r"(?m)^\s{0,3}#{2,6}\s+");
static_regex!(MARKDOWN_HRULE, r"(?m)^\s{0,3}(?:-{3,}|\*{3,})\s*$");
static_regex!(MARKDOWN_BLOCKQUOTE, r"(?m)^\s{0,3}>\s?");
static_regex!(MARKDOWN_BULLET, r"(?m)^\s{0,3}[-*]\s+");
static_regex!(
    MARKDOWN_BOLD_STAR,
    r"(^|[^\w])\*\*(\S(?:.*?\S)?)\*\*([^\w]|$)"
);
static_regex!(
    MARKDOWN_BOLD_UNDERSCORE,
    r"(^|[^\w])__(\S(?:.*?\S)?)__([^\w]|$)"
);
static_regex!(
    MARKDOWN_ITALIC_STAR,
    r"(^|[^\w:/])\*([^\s*_](?:.*?[^\s*_])?)\*([^\w]|$)"
);
static_regex!(
    MARKDOWN_ITALIC_UNDERSCORE,
    r"(^|[^\w:/])_([^\s*_](?:.*?[^\s*_])?)_([^\w]|$)"
);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CleanupStages {
    pub raw: String,
    pub persona: String,
    pub context: String,
    pub identity: String,
    pub final_text: String,
}

#[must_use]
pub fn cleanup_response(
    response: &str,
    contexts: &[Option<String>],
    user_identity: Option<&str>,
) -> CleanupStages {
    let raw = response.to_owned();
    let persona = remove_gordo_prefix(&raw);
    let context = strip_leading_context(&persona, contexts);
    let identity = strip_user_identity_prefix(&context, user_identity);
    let final_text = strip_markdown_formatting(&clean_duplicate_response(&identity));
    CleanupStages {
        raw,
        persona,
        context,
        identity,
        final_text,
    }
}

#[must_use]
pub fn remove_gordo_prefix(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    text.lines()
        .map(|line| regex_replace_all(&GORDO_PREFIX, line, ""))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_owned()
}

#[must_use]
pub fn clean_duplicate_response(response: &str) -> String {
    if response.is_empty() {
        return String::new();
    }
    let mut lines: Vec<&str> = Vec::new();
    for line in response.split('\n') {
        let stripped = line.trim();
        if !stripped.is_empty() && lines.last().copied() != Some(stripped) {
            lines.push(stripped);
        }
    }
    let cleaned = lines.join("\n");
    let mut sentences: Vec<&str> = Vec::new();
    for sentence in cleaned.split(". ") {
        let stripped = sentence.trim();
        if !stripped.is_empty() && sentences.last().copied() != Some(stripped) {
            sentences.push(stripped);
        }
    }
    sentences.join(". ").replace("..", ".")
}

#[must_use]
pub fn strip_leading_context(response: &str, contexts: &[Option<String>]) -> String {
    if response.is_empty() || contexts.is_empty() {
        return response.to_owned();
    }
    let normalized = contexts
        .iter()
        .filter_map(|context| context.as_deref().map(str::trim))
        .filter(|context| !context.is_empty())
        .collect::<Vec<_>>();
    if normalized.is_empty() {
        return response.to_owned();
    }
    let mut trimmed = response.to_owned();
    for _ in 0..normalized.len() {
        let mut changed = false;
        for context in &normalized {
            if starts_with_case_insensitive(&trimmed, context) {
                trimmed = trimmed[context.len()..]
                    .trim_start_matches([' ', '\t', ':', '-', '\n'])
                    .to_owned();
                changed = true;
                break;
            }
        }
        if !changed {
            break;
        }
    }
    trimmed
}

#[must_use]
pub fn strip_user_identity_prefix(response: &str, user_identity: Option<&str>) -> String {
    let Some(identity) = user_identity
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return response.to_owned();
    };
    if response.is_empty() {
        return String::new();
    }
    let leading_trimmed = response.trim_start();
    if !starts_with_case_insensitive(leading_trimmed, identity) {
        return response.to_owned();
    }
    let after_identity = &leading_trimmed[identity.len()..];
    let after_space = after_identity.trim_start();
    let Some(after_colon) = after_space.strip_prefix(':') else {
        return response.to_owned();
    };
    after_colon.trim_start().to_owned()
}

#[must_use]
pub fn strip_markdown_formatting(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let mut cleaned = regex_replace_all(&MARKDOWN_CODE_FENCE, text, "$1");
    cleaned = regex_replace_all(&MARKDOWN_IMAGE, &cleaned, "$1");
    cleaned = regex_replace_all(&MARKDOWN_LINK, &cleaned, "$1");
    cleaned = regex_replace_all(&MARKDOWN_INLINE_CODE, &cleaned, "$1");
    cleaned = regex_replace_all(&MARKDOWN_HEADER, &cleaned, "");
    cleaned = regex_replace_all(&MARKDOWN_HRULE, &cleaned, "");
    cleaned = regex_replace_all(&MARKDOWN_BLOCKQUOTE, &cleaned, "");
    cleaned = regex_replace_all(&MARKDOWN_BULLET, &cleaned, "");
    cleaned = replace_until_stable(&MARKDOWN_BOLD_STAR, &cleaned, "$1$2$3");
    cleaned = replace_until_stable(&MARKDOWN_BOLD_UNDERSCORE, &cleaned, "$1$2$3");
    cleaned = replace_until_stable(&MARKDOWN_ITALIC_STAR, &cleaned, "$1$2$3");
    cleaned = replace_until_stable(&MARKDOWN_ITALIC_UNDERSCORE, &cleaned, "$1$2$3");
    cleaned
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_owned()
}

fn regex_replace_all(pattern: &Option<Regex>, text: &str, replacement: &str) -> String {
    pattern.as_ref().map_or_else(
        || text.to_owned(),
        |pattern| pattern.replace_all(text, replacement).into_owned(),
    )
}

fn replace_until_stable(pattern: &Option<Regex>, text: &str, replacement: &str) -> String {
    let mut current = text.to_owned();
    loop {
        let next = regex_replace_all(pattern, &current, replacement);
        if next == current {
            return current;
        }
        current = next;
    }
}

fn starts_with_case_insensitive(value: &str, prefix: &str) -> bool {
    value
        .get(..prefix.len())
        .is_some_and(|candidate| candidate.to_lowercase() == prefix.to_lowercase())
}

#[cfg(test)]
mod tests {
    use super::{
        CleanupStages, clean_duplicate_response, cleanup_response, remove_gordo_prefix,
        strip_leading_context, strip_markdown_formatting, strip_user_identity_prefix,
    };

    #[test]
    fn removes_persona_prefixes_duplicates_context_and_identity() {
        assert_eq!(
            remove_gordo_prefix("Gordo: hola\n  gordo : mundo"),
            "hola\nmundo"
        );
        assert_eq!(
            clean_duplicate_response(" hola \nhola\nmundo. mundo. final.."),
            "hola\nmundo. mundo. final."
        );
        assert_eq!(
            strip_leading_context(
                "Contexto uno: contexto dos - respuesta",
                &[
                    Some("contexto uno".to_owned()),
                    Some("contexto dos".to_owned())
                ],
            ),
            "respuesta"
        );
        assert_eq!(
            strip_user_identity_prefix("  @User : respuesta", Some("@user")),
            "respuesta"
        );
        assert_eq!(
            strip_user_identity_prefix("@user sin dos puntos", Some("@user")),
            "@user sin dos puntos"
        );
    }

    #[test]
    fn strips_supported_markdown_without_changing_url_underscores() {
        let input = concat!(
            "## Header\n",
            "> **bold** and *italic* and __strong__ and _soft_\n",
            "- `code` [link](https://example.com) ![cat](cat.png)\n",
            "---\n",
            "https://example.com/my_path_value\n",
            "```rust\nlet x = 1;\n```"
        );
        assert_eq!(
            strip_markdown_formatting(input),
            concat!(
                "Header\n",
                "bold and italic and strong and soft\n",
                "code link cat\n",
                "https://example.com/my_path_value\n",
                "let x = 1;"
            )
        );
    }

    #[test]
    fn returns_all_cleanup_stages() {
        assert_eq!(
            cleanup_response(
                "Gordo: CONTEXT: @User: **answer**\nanswer",
                &[Some("context".to_owned())],
                Some("@user"),
            ),
            CleanupStages {
                raw: "Gordo: CONTEXT: @User: **answer**\nanswer".to_owned(),
                persona: "CONTEXT: @User: **answer**\nanswer".to_owned(),
                context: "@User: **answer**\nanswer".to_owned(),
                identity: "**answer**\nanswer".to_owned(),
                final_text: "answer\nanswer".to_owned(),
            }
        );
    }

    #[test]
    fn empty_and_missing_inputs_are_identities() {
        assert_eq!(remove_gordo_prefix(""), "");
        assert_eq!(clean_duplicate_response(""), "");
        assert_eq!(strip_markdown_formatting(""), "");
        assert_eq!(strip_leading_context("answer", &[]), "answer");
        assert_eq!(strip_user_identity_prefix("answer", None), "answer");
    }
}
