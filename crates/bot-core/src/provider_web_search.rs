//! Web-search accounting and grounding policy for provider tool rounds.

use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebSearchRoundMetrics {
    pub metadata_request_count: Option<i64>,
    pub citation_count: usize,
    pub grounded: Option<bool>,
    pub request_count: usize,
}

#[must_use]
pub fn nonnegative_limit(value: Option<&Value>) -> u64 {
    python_int(value).map_or(0, |number| number.max(0) as u64)
}

#[must_use]
pub fn round_metrics(
    server_request_value: Option<&Value>,
    tool_names: &[String],
    annotation_types: &[String],
) -> WebSearchRoundMetrics {
    let server_count = python_int(server_request_value);
    let direct_count = tool_names
        .iter()
        .filter(|name| name.as_str() == "web_search")
        .count();
    let citation_count = annotation_types
        .iter()
        .filter(|annotation_type| annotation_type.as_str() == "url_citation")
        .count();

    let mut metadata_request_count = server_count;
    if direct_count > 0 {
        metadata_request_count = Some(saturating_i64(direct_count));
    } else if metadata_request_count.is_none() && citation_count > 0 {
        metadata_request_count = Some(1);
    }
    let grounded = metadata_request_count
        .filter(|count| *count > 0)
        .map(|_| citation_count > 0);

    let positive_server_count = server_count.unwrap_or(0).max(0) as u64;
    let request_count = if positive_server_count > 0 {
        saturating_usize(positive_server_count)
    } else if direct_count > 0 {
        direct_count
    } else {
        usize::from(citation_count > 0)
    };

    WebSearchRoundMetrics {
        metadata_request_count,
        citation_count,
        grounded,
        request_count,
    }
}

#[must_use]
pub fn remaining_budget(remaining: Option<usize>, request_count: usize) -> Option<usize> {
    remaining.map(|value| value.saturating_sub(request_count))
}

pub fn source_urls(messages: &Value) -> Result<Vec<String>, &'static str> {
    let messages = messages
        .as_array()
        .ok_or("provider messages must be a JSON array")?;
    let mut web_search_call_ids = HashSet::new();
    for message in messages {
        let Some(message) = message.as_object() else {
            continue;
        };
        if string_field(message.get("role")) != "assistant" {
            continue;
        }
        let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) else {
            continue;
        };
        for tool_call in tool_calls {
            let Some(tool_call) = tool_call.as_object() else {
                continue;
            };
            let name = tool_call
                .get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("name"));
            if string_field(name) != "web_search" {
                continue;
            }
            let call_id = string_field(tool_call.get("id"));
            if !call_id.is_empty() {
                web_search_call_ids.insert(call_id.to_owned());
            }
        }
    }

    let mut urls = Vec::new();
    for message in messages {
        let Some(message) = message.as_object() else {
            continue;
        };
        if string_field(message.get("role")) != "tool"
            || !web_search_call_ids.contains(string_field(message.get("tool_call_id")))
        {
            continue;
        }
        let Some(content) = message.get("content").and_then(Value::as_str) else {
            continue;
        };
        let Ok(payload) = serde_json::from_str::<Value>(content) else {
            continue;
        };
        let Some(results) = payload
            .as_object()
            .and_then(|value| value.get("results"))
            .and_then(Value::as_array)
        else {
            continue;
        };
        for result in results {
            let Some(url) = result
                .as_object()
                .and_then(|value| value.get("url"))
                .and_then(Value::as_str)
            else {
                continue;
            };
            let url = url.trim_end_matches(['.', ',', ')', ';', ']']);
            if !url.is_empty() && !urls.iter().any(|existing| existing == url) {
                urls.push(url.to_owned());
            }
        }
    }
    Ok(urls)
}

#[must_use]
pub fn outcome_is_grounded(source_count: usize, citation_count: usize, text: &str) -> bool {
    !text.trim().is_empty() && (source_count > 0 || citation_count > 0)
}

fn python_int(value: Option<&Value>) -> Option<i64> {
    match value? {
        Value::Null => Some(0),
        Value::Bool(value) => Some(i64::from(*value)),
        Value::Number(value) => value
            .as_i64()
            .or_else(|| value.as_u64().and_then(|number| i64::try_from(number).ok()))
            .or_else(|| value.as_f64().and_then(float_to_i64)),
        Value::String(value) => value.trim().parse::<i64>().ok(),
        Value::Array(_) | Value::Object(_) => None,
    }
}

fn float_to_i64(value: f64) -> Option<i64> {
    if !value.is_finite() || value < i64::MIN as f64 || value > i64::MAX as f64 {
        return None;
    }
    Some(value.trunc() as i64)
}

fn string_field(value: Option<&Value>) -> &str {
    value.and_then(Value::as_str).unwrap_or("")
}

fn saturating_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn saturating_usize(value: u64) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::{
        WebSearchRoundMetrics, nonnegative_limit, outcome_is_grounded, remaining_budget,
        round_metrics, source_urls,
    };
    use serde_json::{Value, json};

    #[test]
    fn normalizes_limits_with_python_integer_semantics() {
        for (value, expected) in [
            (json!(3), 3),
            (json!(" 4 "), 4),
            (json!(2.9), 2),
            (json!(true), 1),
            (json!(-8), 0),
            (json!("invalid"), 0),
            (json!([2]), 0),
            (Value::Null, 0),
        ] {
            assert_eq!(nonnegative_limit(Some(&value)), expected);
        }
        assert_eq!(nonnegative_limit(None), 0);
    }

    #[test]
    fn applies_server_direct_and_citation_precedence() {
        let cases = [
            (
                Some(json!(2)),
                vec![],
                vec![],
                WebSearchRoundMetrics {
                    metadata_request_count: Some(2),
                    citation_count: 0,
                    grounded: Some(false),
                    request_count: 2,
                },
            ),
            (
                Some(json!(7)),
                vec!["web_search".to_owned(), "web_search".to_owned()],
                vec!["url_citation".to_owned()],
                WebSearchRoundMetrics {
                    metadata_request_count: Some(2),
                    citation_count: 1,
                    grounded: Some(true),
                    request_count: 7,
                },
            ),
            (
                None,
                vec!["calculate".to_owned()],
                vec!["url_citation".to_owned()],
                WebSearchRoundMetrics {
                    metadata_request_count: Some(1),
                    citation_count: 1,
                    grounded: Some(true),
                    request_count: 1,
                },
            ),
            (
                Some(json!(-2)),
                vec![],
                vec!["other".to_owned()],
                WebSearchRoundMetrics {
                    metadata_request_count: Some(-2),
                    citation_count: 0,
                    grounded: None,
                    request_count: 0,
                },
            ),
        ];
        for (server, tools, annotations, expected) in cases {
            assert_eq!(
                round_metrics(server.as_ref(), &tools, &annotations),
                expected
            );
        }
    }

    #[test]
    fn updates_bounded_and_unbounded_budgets() {
        assert_eq!(remaining_budget(Some(3), 2), Some(1));
        assert_eq!(remaining_budget(Some(1), 9), Some(0));
        assert_eq!(remaining_budget(None, 9), None);
    }

    #[test]
    fn extracts_only_deduplicated_search_result_urls() {
        let messages = json!([
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "search", "function": {"name": "web_search"}},
                    {"id": "fetch", "function": {"name": "web_fetch"}}
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "search",
                "content": "{\"results\":[{\"url\":\"https://one.example/a).\"},{\"url\":\"https://two.example\"},{\"url\":\"https://two.example\"}]}"
            },
            {
                "role": "tool",
                "tool_call_id": "fetch",
                "content": "{\"results\":[{\"url\":\"https://ignored.example\"}]}"
            },
            {"role": "tool", "tool_call_id": "search", "content": "invalid"}
        ]);
        assert_eq!(
            source_urls(&messages),
            Ok(vec![
                "https://one.example/a".to_owned(),
                "https://two.example".to_owned()
            ])
        );
        assert!(source_urls(&json!({})).is_err());
    }

    #[test]
    fn grounds_only_nonempty_answers_with_evidence() {
        assert!(outcome_is_grounded(1, 0, "answer"));
        assert!(outcome_is_grounded(0, 1, " answer "));
        assert!(!outcome_is_grounded(0, 0, "answer"));
        assert!(!outcome_is_grounded(1, 1, "  "));
    }
}
