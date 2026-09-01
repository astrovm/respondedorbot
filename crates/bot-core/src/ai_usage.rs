//! Durable provider-call identity and interrupted-usage policy.

use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderSegmentIdentity<'a> {
    pub source_with_default: &'a str,
    pub source_or_provider: &'a str,
    pub kind_or_unknown: &'a str,
    pub model_or_unknown: &'a str,
    pub provider_generation_id: Option<&'a str>,
    pub provider_request_id: Option<&'a str>,
    pub tool_rounds: Option<&'a str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderUsageStatus<'a> {
    pub source: &'a str,
    pub stream_interrupted: bool,
    pub provider_usage_pending: bool,
    pub cost_is_positive: bool,
}

/// Preserve provider retry identities across restarts and language versions.
#[must_use]
pub fn provider_segment_id(
    identity: &ProviderSegmentIdentity<'_>,
    python_canonical_json: &str,
) -> String {
    if let Some(provider_id) = identity
        .provider_generation_id
        .filter(|value| !value.is_empty())
        .or_else(|| {
            identity
                .provider_request_id
                .filter(|value| !value.is_empty())
        })
    {
        return format!("{}:{provider_id}", identity.source_with_default);
    }
    if let Some(tool_rounds) = identity.tool_rounds.filter(|value| !value.is_empty()) {
        return format!(
            "{}:{}:{}:{tool_rounds}",
            identity.source_or_provider, identity.kind_or_unknown, identity.model_or_unknown,
        );
    }
    Sha256::digest(python_canonical_json.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Derive the durable identity for one normalized provider-usage segment.
#[must_use]
pub fn stable_provider_segment_id(segment: &Value) -> String {
    let metadata = segment.get("metadata").and_then(Value::as_object);
    let source = segment
        .get("source")
        .and_then(Value::as_str)
        .unwrap_or("provider");
    let kind = segment
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let model = segment
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let canonical = serde_json::to_string(segment).unwrap_or_default();
    provider_segment_id(
        &ProviderSegmentIdentity {
            source_with_default: source,
            source_or_provider: source,
            kind_or_unknown: kind,
            model_or_unknown: model,
            provider_generation_id: metadata
                .and_then(|value| value.get("provider_generation_id"))
                .and_then(Value::as_str),
            provider_request_id: metadata
                .and_then(|value| value.get("provider_request_id"))
                .and_then(Value::as_str),
            tool_rounds: metadata
                .and_then(|value| value.get("tool_rounds"))
                .and_then(Value::as_str),
        },
        &canonical,
    )
}

/// Interrupted OpenRouter calls need reconciliation until positive cost arrives.
#[must_use]
pub fn needs_reconciliation(status: ProviderUsageStatus<'_>) -> bool {
    status.source == "openrouter"
        && (status.stream_interrupted || status.provider_usage_pending)
        && !status.cost_is_positive
}

#[cfg(test)]
mod tests {
    use super::{
        ProviderSegmentIdentity, ProviderUsageStatus, needs_reconciliation, provider_segment_id,
        stable_provider_segment_id,
    };

    fn identity<'a>() -> ProviderSegmentIdentity<'a> {
        ProviderSegmentIdentity {
            source_with_default: "openrouter",
            source_or_provider: "openrouter",
            kind_or_unknown: "chat",
            model_or_unknown: "test/model",
            provider_generation_id: None,
            provider_request_id: None,
            tool_rounds: None,
        }
    }

    #[test]
    fn prefers_generation_then_request_then_tool_round_identity() {
        let mut input = identity();
        input.provider_generation_id = Some("generation-1");
        input.provider_request_id = Some("request-1");
        input.tool_rounds = Some("2");
        assert_eq!(provider_segment_id(&input, "{}"), "openrouter:generation-1");

        input.provider_generation_id = Some("");
        assert_eq!(provider_segment_id(&input, "{}"), "openrouter:request-1");

        input.provider_request_id = None;
        assert_eq!(
            provider_segment_id(&input, "{}"),
            "openrouter:chat:test/model:2"
        );
    }

    #[test]
    fn hashes_the_exact_python_canonical_payload_as_the_compatibility_fallback() {
        let canonical = "{\"kind\": \"chat\", \"metadata\": {}, \"model\": \"m\", \
            \"source\": \"openrouter\", \"usage\": {\"cost\": 0.1}}";
        assert_eq!(
            provider_segment_id(&identity(), canonical),
            "9720f6f861361d9e0abf8de652c4438e8872c5b593d0fda29dbcaf678b5240ad"
        );
    }

    #[test]
    fn stable_segment_identity_prefers_provider_ids_and_hashes_tool_segments() {
        assert_eq!(
            stable_provider_segment_id(&serde_json::json!({
                "kind": "chat",
                "model": "test/model",
                "source": "openrouter",
                "metadata": {"provider_generation_id": "generation-1"}
            })),
            "openrouter:generation-1"
        );
        let first = stable_provider_segment_id(&serde_json::json!({
            "kind": "web_search",
            "source": "firecrawl",
            "metadata": {"tool_call_id": "call-1", "firecrawl_credits_used": 2}
        }));
        let second = stable_provider_segment_id(&serde_json::json!({
            "kind": "web_search",
            "source": "firecrawl",
            "metadata": {"tool_call_id": "call-2", "firecrawl_credits_used": 2}
        }));
        assert_ne!(first, second);
    }

    #[test]
    fn reconciles_only_pending_openrouter_usage_without_positive_cost() {
        for (status, expected) in [
            (
                ProviderUsageStatus {
                    source: "openrouter",
                    stream_interrupted: true,
                    provider_usage_pending: false,
                    cost_is_positive: false,
                },
                true,
            ),
            (
                ProviderUsageStatus {
                    source: "openrouter",
                    stream_interrupted: false,
                    provider_usage_pending: true,
                    cost_is_positive: false,
                },
                true,
            ),
            (
                ProviderUsageStatus {
                    source: "openrouter",
                    stream_interrupted: true,
                    provider_usage_pending: true,
                    cost_is_positive: true,
                },
                false,
            ),
            (
                ProviderUsageStatus {
                    source: "groq",
                    stream_interrupted: true,
                    provider_usage_pending: true,
                    cost_is_positive: false,
                },
                false,
            ),
        ] {
            assert_eq!(needs_reconciliation(status), expected);
        }
    }
}
