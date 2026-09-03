//! Pure retry and billable-activity decisions for provider completions.

use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderExceptionFacts {
    pub json_decode_error: bool,
    pub connection_error: bool,
    pub timeout_error: bool,
    pub rate_limit_error: bool,
    pub api_status_code: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FinishResponseFacts<'a> {
    pub has_content: bool,
    pub tool_call_count: usize,
    pub has_usage: bool,
    pub finish_reason: Option<&'a str>,
    pub error_status_code: i64,
    pub error_type: &'a str,
}

#[must_use]
pub fn is_retryable_provider_exception(facts: ProviderExceptionFacts) -> bool {
    facts.json_decode_error
        || facts.connection_error
        || facts.timeout_error
        || facts.rate_limit_error
        || facts
            .api_status_code
            .is_some_and(|status| matches!(status, 408 | 409 | 425 | 429) || status >= 500)
}

#[must_use]
pub fn response_has_billable_usage(usage: &Map<String, Value>) -> bool {
    for key in [
        "cost",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    ] {
        if positive_float(usage.get(key)) {
            return true;
        }
    }
    usage
        .get("server_tool_use")
        .and_then(Value::as_object)
        .and_then(|server_tool_use| server_tool_use.get("web_search_requests"))
        .is_some_and(positive_integer)
}

#[must_use]
pub fn is_retryable_finish_response(facts: FinishResponseFacts<'_>) -> bool {
    if facts.has_content || facts.tool_call_count > 0 || facts.has_usage {
        return false;
    }
    let Some(finish_reason) = facts.finish_reason else {
        return true;
    };
    if finish_reason != "error" {
        return false;
    }
    matches!(facts.error_status_code, 408 | 409 | 429)
        || facts.error_status_code >= 500
        || matches!(
            facts.error_type,
            "rate_limit_exceeded"
                | "provider_overloaded"
                | "provider_unavailable"
                | "server"
                | "timeout"
        )
}

#[must_use]
pub fn retry_wait_seconds(attempt: u32) -> Option<u64> {
    2_u64.checked_pow(attempt)
}

fn positive_float(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) | Some(Value::Bool(false)) => false,
        Some(Value::Bool(true)) => true,
        Some(Value::Number(value)) => value.as_f64().is_some_and(|value| value > 0.0),
        Some(Value::String(value)) => value
            .parse::<f64>()
            .is_ok_and(|value| value.is_finite() && value > 0.0),
        Some(Value::Array(_) | Value::Object(_)) => false,
    }
}

fn positive_integer(value: &Value) -> bool {
    match value {
        Value::Bool(value) => *value,
        Value::Number(value) => {
            value.as_i64().is_some_and(|value| value > 0)
                || value.as_u64().is_some_and(|value| value > 0)
                || value.as_f64().is_some_and(|value| value.trunc() > 0.0)
        }
        Value::String(value) => value.parse::<i64>().is_ok_and(|value| value > 0),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, json};

    use super::{
        FinishResponseFacts, ProviderExceptionFacts, is_retryable_finish_response,
        is_retryable_provider_exception, response_has_billable_usage, retry_wait_seconds,
    };

    #[test]
    fn classifies_transient_exception_facts_and_bounded_delays() {
        for facts in [
            ProviderExceptionFacts {
                json_decode_error: true,
                connection_error: false,
                timeout_error: false,
                rate_limit_error: false,
                api_status_code: None,
            },
            ProviderExceptionFacts {
                json_decode_error: false,
                connection_error: true,
                timeout_error: false,
                rate_limit_error: false,
                api_status_code: None,
            },
            ProviderExceptionFacts {
                json_decode_error: false,
                connection_error: false,
                timeout_error: true,
                rate_limit_error: false,
                api_status_code: None,
            },
            ProviderExceptionFacts {
                json_decode_error: false,
                connection_error: false,
                timeout_error: false,
                rate_limit_error: true,
                api_status_code: None,
            },
            ProviderExceptionFacts {
                json_decode_error: false,
                connection_error: false,
                timeout_error: false,
                rate_limit_error: false,
                api_status_code: Some(503),
            },
            ProviderExceptionFacts {
                json_decode_error: false,
                connection_error: false,
                timeout_error: false,
                rate_limit_error: false,
                api_status_code: Some(408),
            },
            ProviderExceptionFacts {
                json_decode_error: false,
                connection_error: false,
                timeout_error: false,
                rate_limit_error: false,
                api_status_code: Some(425),
            },
        ] {
            assert!(is_retryable_provider_exception(facts));
        }
        assert!(!is_retryable_provider_exception(ProviderExceptionFacts {
            json_decode_error: false,
            connection_error: false,
            timeout_error: false,
            rate_limit_error: false,
            api_status_code: Some(400),
        }));
        assert_eq!(retry_wait_seconds(0), Some(1));
        assert_eq!(retry_wait_seconds(1), Some(2));
        assert_eq!(retry_wait_seconds(64), None);
    }

    #[test]
    fn detects_every_billable_usage_shape_and_ignores_invalid_values() {
        for usage in [
            json!({"cost": "0.1"}),
            json!({"prompt_tokens": 1}),
            json!({"completion_tokens": 1.9}),
            json!({"total_tokens": true}),
            json!({"input_tokens": 1}),
            json!({"output_tokens": 1}),
            json!({"server_tool_use": {"web_search_requests": "2"}}),
        ] {
            assert!(response_has_billable_usage(
                usage.as_object().unwrap_or(&Map::new())
            ));
        }
        for usage in [
            json!({}),
            json!({"cost": "invalid"}),
            json!({"prompt_tokens": -1}),
            json!({"server_tool_use": {"web_search_requests": "invalid"}}),
        ] {
            assert!(!response_has_billable_usage(
                usage.as_object().unwrap_or(&Map::new())
            ));
        }
    }

    #[test]
    fn retries_only_empty_transient_finish_responses() {
        let baseline = FinishResponseFacts {
            has_content: false,
            tool_call_count: 0,
            has_usage: false,
            finish_reason: None,
            error_status_code: 0,
            error_type: "",
        };
        assert!(is_retryable_finish_response(baseline));
        assert!(is_retryable_finish_response(FinishResponseFacts {
            finish_reason: Some("error"),
            error_status_code: 503,
            ..baseline
        }));
        assert!(is_retryable_finish_response(FinishResponseFacts {
            finish_reason: Some("error"),
            error_type: "provider_unavailable",
            ..baseline
        }));
        for facts in [
            FinishResponseFacts {
                has_content: true,
                ..baseline
            },
            FinishResponseFacts {
                tool_call_count: 1,
                ..baseline
            },
            FinishResponseFacts {
                has_usage: true,
                ..baseline
            },
            FinishResponseFacts {
                finish_reason: Some("stop"),
                ..baseline
            },
            FinishResponseFacts {
                finish_reason: Some("error"),
                error_status_code: 400,
                error_type: "invalid_request",
                ..baseline
            },
        ] {
            assert!(!is_retryable_finish_response(facts));
        }
    }
}
