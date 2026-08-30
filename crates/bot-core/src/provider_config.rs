//! Provider credential, availability, and server-tool configuration policy.

use serde::Serialize;

pub const DEFAULT_OPENROUTER_URL: &str = "https://openrouter.ai/api/v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WebSearchTool {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub parameters: WebSearchParameters,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WebSearchParameters {
    pub engine: &'static str,
    pub max_results: i64,
    pub max_uses: i64,
    pub max_total_results: i128,
}

#[must_use]
pub fn clean_value(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

#[must_use]
pub fn groq_api_key(
    account: &str,
    free_api_key: Option<&str>,
    paid_api_key: Option<&str>,
) -> Option<String> {
    clean_value(if account == "free" {
        free_api_key
    } else {
        paid_api_key
    })
}

#[must_use]
pub fn configured_accounts(account_order: &[String], configured: &[bool]) -> Vec<String> {
    account_order
        .iter()
        .zip(configured)
        .filter(|(_, is_configured)| **is_configured)
        .map(|(account, _)| account.clone())
        .collect()
}

#[must_use]
pub fn groq_backoff_key(account: &str, scope: &str) -> String {
    format!("groq:{account}:{scope}").to_lowercase()
}

#[must_use]
pub fn scope_is_available(backoff_active: &[bool]) -> bool {
    backoff_active.is_empty() || backoff_active.iter().any(|active| !active)
}

#[must_use]
pub fn web_search_tool(max_results: i64, max_queries: i64) -> WebSearchTool {
    WebSearchTool {
        kind: "openrouter:web_search",
        parameters: WebSearchParameters {
            engine: "firecrawl",
            max_results,
            max_uses: max_queries,
            max_total_results: i128::from(max_results) * i128::from(max_queries),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_OPENROUTER_URL, WebSearchParameters, WebSearchTool, clean_value,
        configured_accounts, groq_api_key, groq_backoff_key, scope_is_available, web_search_tool,
    };

    #[test]
    fn credentials_are_trimmed_and_selected_by_legacy_account_rules() {
        assert_eq!(clean_value(Some("  key  ")), Some("key".to_owned()));
        assert_eq!(clean_value(Some(" \t ")), None);
        assert_eq!(clean_value(None), None);
        assert_eq!(
            groq_api_key("free", Some(" free-key "), Some("paid-key")),
            Some("free-key".to_owned()),
        );
        assert_eq!(
            groq_api_key("paid", Some("free-key"), Some(" paid-key ")),
            Some("paid-key".to_owned()),
        );
        assert_eq!(groq_api_key("unknown", None, Some("")), None);
        assert_eq!(DEFAULT_OPENROUTER_URL, "https://openrouter.ai/api/v1");
    }

    #[test]
    fn account_order_backoff_keys_and_scope_availability_are_stable() {
        let accounts = vec!["free".to_owned(), "paid".to_owned(), "later".to_owned()];
        assert_eq!(
            configured_accounts(&accounts, &[true, false, true]),
            vec!["free".to_owned(), "later".to_owned()],
        );
        assert_eq!(
            configured_accounts(&accounts, &[true]),
            vec!["free".to_owned()],
        );
        assert_eq!(groq_backoff_key("FREE", "CHAT"), "groq:free:chat");
        assert!(scope_is_available(&[]));
        assert!(scope_is_available(&[true, false]));
        assert!(!scope_is_available(&[true, true]));
    }

    #[test]
    fn web_search_tool_preserves_limits_and_uses_wide_multiplication() {
        assert_eq!(
            web_search_tool(i64::MAX, 2),
            WebSearchTool {
                kind: "openrouter:web_search",
                parameters: WebSearchParameters {
                    engine: "firecrawl",
                    max_results: i64::MAX,
                    max_uses: 2,
                    max_total_results: i128::from(i64::MAX) * 2,
                },
            },
        );
        assert_eq!(web_search_tool(-2, 3).parameters.max_total_results, -6);
    }
}
