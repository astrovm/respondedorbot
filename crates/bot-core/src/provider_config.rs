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
        DEFAULT_OPENROUTER_URL, WebSearchParameters, WebSearchTool, clean_value, web_search_tool,
    };

    #[test]
    fn credentials_are_trimmed_and_blank_values_are_ignored() {
        assert_eq!(clean_value(Some("  key  ")), Some("key".to_owned()));
        assert_eq!(clean_value(Some(" \t ")), None);
        assert_eq!(clean_value(None), None);
        assert_eq!(DEFAULT_OPENROUTER_URL, "https://openrouter.ai/api/v1");
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
