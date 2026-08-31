//! Native Firecrawl AI tool execution and usage accounting.

use std::time::Duration;

use bot_adapters::firecrawl::{FirecrawlTransport, SearchOutcome, search_with};
use bot_core::locale::Locale;
use serde_json::{Value, json};

use crate::chat_tool_loop::ToolExecutionResult;
use crate::tool_requests::{ExternalToolExecutor, ExternalToolRequest};

pub struct FirecrawlTool<Transport, Sleep> {
    transport: Transport,
    sleep: Sleep,
    api_key: String,
    locale: Locale,
}

impl<Transport, Sleep> FirecrawlTool<Transport, Sleep> {
    #[must_use]
    pub fn new(transport: Transport, sleep: Sleep, api_key: &str, locale: Locale) -> Self {
        Self {
            transport,
            sleep,
            api_key: api_key.to_owned(),
            locale,
        }
    }
}

impl<Transport, Sleep> ExternalToolExecutor for FirecrawlTool<Transport, Sleep>
where
    Transport: FirecrawlTransport,
    Sleep: Fn(Duration),
{
    fn execute(&mut self, request: ExternalToolRequest, tool_call_id: &str) -> ToolExecutionResult {
        let ExternalToolRequest::WebSearch { query } = request else {
            return ToolExecutionResult::output("web_search received an incompatible request");
        };
        match search_with(&self.transport, &self.api_key, &query, &self.sleep) {
            Ok(outcome) => outcome_result(outcome, &query, tool_call_id, self.locale),
            Err(error) => ToolExecutionResult::output(format!("Tool 'web_search' error: {error}")),
        }
    }
}

fn outcome_result(
    outcome: SearchOutcome,
    query: &str,
    tool_call_id: &str,
    locale: Locale,
) -> ToolExecutionResult {
    match outcome {
        SearchOutcome::Success {
            results,
            credits_used,
            request_id: _,
            query: _,
        } => {
            let billing_segment = positive_credits(&credits_used).map(|credits_used| {
                json!({
                    "kind": "web_search",
                    "model": "",
                    "usage": {},
                    "source": "firecrawl",
                    "metadata": {
                        "provider": "firecrawl",
                        "tool_call_id": tool_call_id,
                        "web_search_requests": 1,
                        "firecrawl_credits_used": credits_used,
                    }
                })
            });
            ToolExecutionResult {
                output: json!({"query": query, "results": results}).to_string(),
                billing_segment,
            }
        }
        SearchOutcome::Timeout => ToolExecutionResult::output(localized(
            locale,
            "Error de búsqueda: Firecrawl agotó el tiempo de espera.",
            "Search error: Firecrawl timed out.",
        )),
        SearchOutcome::Connection => ToolExecutionResult::output(localized(
            locale,
            "Error de búsqueda: no se pudo conectar con Firecrawl.",
            "Search error: could not connect to Firecrawl.",
        )),
        SearchOutcome::InvalidJson => ToolExecutionResult::output(localized(
            locale,
            "Error de búsqueda: Firecrawl devolvió JSON inválido.",
            "Search error: Firecrawl returned invalid JSON.",
        )),
        SearchOutcome::HttpError {
            status_code,
            detail,
        } => ToolExecutionResult::output(search_error(
            locale,
            &format!("Firecrawl HTTP {status_code}: {detail}"),
        )),
        SearchOutcome::ApiError { detail } => {
            ToolExecutionResult::output(search_error(locale, &detail))
        }
    }
}

fn positive_credits(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_str()?.parse().ok())
        .filter(|credits| *credits > 0)
}

fn localized(locale: Locale, spanish: &str, english: &str) -> String {
    match locale {
        Locale::Es => spanish.to_owned(),
        Locale::En => english.to_owned(),
    }
}

fn search_error(locale: Locale, detail: &str) -> String {
    match locale {
        Locale::Es => format!("Error de búsqueda: {detail}"),
        Locale::En => format!("Search error: {detail}"),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use bot_adapters::firecrawl::{HttpResponse, SearchRequest, TransportError};

    use super::*;

    struct Transport {
        responses: RefCell<Vec<Result<HttpResponse, TransportError>>>,
    }

    impl FirecrawlTransport for Transport {
        fn post(&self, _request: &SearchRequest) -> Result<HttpResponse, TransportError> {
            self.responses.borrow_mut().remove(0)
        }
    }

    fn tool(
        responses: Vec<Result<HttpResponse, TransportError>>,
        locale: Locale,
    ) -> FirecrawlTool<Transport, impl Fn(Duration)> {
        FirecrawlTool::new(
            Transport {
                responses: RefCell::new(responses),
            },
            |_| {},
            "synthetic-key",
            locale,
        )
    }

    #[test]
    fn success_returns_bounded_sources_and_a_billable_firecrawl_segment() {
        let mut tool = tool(
            vec![Ok(HttpResponse {
                status_code: 200,
                body: json!({
                    "success": true,
                    "creditsUsed": 2,
                    "id": "request-1",
                    "data": {"web": [{
                        "title": "Synthetic",
                        "url": "https://example.test",
                        "description": "evidence"
                    }]}
                })
                .to_string(),
            })],
            Locale::En,
        );
        let result = tool.execute(
            ExternalToolRequest::WebSearch {
                query: "synthetic query".to_owned(),
            },
            "call-1",
        );
        let output: Value = serde_json::from_str(&result.output).unwrap_or(Value::Null);
        assert_eq!(output["query"], "synthetic query");
        assert_eq!(output["results"][0]["url"], "https://example.test");
        let segment = result.billing_segment.unwrap_or(Value::Null);
        assert_eq!(segment["kind"], "web_search");
        assert_eq!(segment["metadata"]["tool_call_id"], "call-1");
        assert_eq!(segment["metadata"]["firecrawl_credits_used"], 2);
    }

    #[test]
    fn zero_usage_and_all_provider_failures_have_safe_localized_results() {
        let outcomes = [
            (
                HttpResponse {
                    status_code: 200,
                    body: json!({"success": true, "creditsUsed": 0, "data": []}).to_string(),
                },
                "{\"query\":\"q\",\"results\":[]}",
            ),
            (
                HttpResponse {
                    status_code: 200,
                    body: "not-json".to_owned(),
                },
                "Search error: Firecrawl returned invalid JSON.",
            ),
            (
                HttpResponse {
                    status_code: 400,
                    body: json!({"error": "bad query"}).to_string(),
                },
                "Search error: Firecrawl HTTP 400: bad query",
            ),
            (
                HttpResponse {
                    status_code: 200,
                    body: json!({"success": false, "error": "api rejected"}).to_string(),
                },
                "Search error: api rejected",
            ),
        ];
        for (response, expected) in outcomes {
            let mut tool = tool(vec![Ok(response)], Locale::En);
            let result = tool.execute(
                ExternalToolRequest::WebSearch {
                    query: "q".to_owned(),
                },
                "call",
            );
            assert_eq!(result.output, expected);
            assert!(result.billing_segment.is_none());
        }

        for (error, expected) in [
            (
                TransportError::Timeout,
                "Error de búsqueda: Firecrawl agotó el tiempo de espera.",
            ),
            (
                TransportError::Connection,
                "Error de búsqueda: no se pudo conectar con Firecrawl.",
            ),
        ] {
            let mut tool = tool(
                vec![Err(error.clone()), Err(error.clone()), Err(error)],
                Locale::Es,
            );
            assert_eq!(
                tool.execute(
                    ExternalToolRequest::WebSearch {
                        query: "q".to_owned()
                    },
                    "call"
                )
                .output,
                expected
            );
        }
    }

    #[test]
    fn incompatible_requests_and_nonretryable_transport_errors_are_explicit() {
        let mut incompatible_tool = tool(Vec::new(), Locale::En);
        assert_eq!(
            incompatible_tool
                .execute(
                    ExternalToolRequest::Weather {
                        location: "Synthetic City".to_owned()
                    },
                    "call"
                )
                .output,
            "web_search received an incompatible request"
        );
        let mut tool = tool(
            vec![Err(TransportError::Other("synthetic failure".to_owned()))],
            Locale::En,
        );
        assert_eq!(
            tool.execute(
                ExternalToolRequest::WebSearch {
                    query: "q".to_owned()
                },
                "call"
            )
            .output,
            "Tool 'web_search' error: Firecrawl transport failed: synthetic failure"
        );
    }
}
