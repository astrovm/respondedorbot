//! Typed normalization of provider response identity and routing metadata.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderUsageFacts<'a> {
    pub requested_model: &'a str,
    pub response_model: Option<&'a str>,
    pub upstream_provider: Option<&'a str>,
    pub service_tier: Option<&'a str>,
    pub source: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedProviderUsage {
    pub model: String,
    pub requested_model_metadata: Option<String>,
    pub upstream_provider: Option<String>,
    pub service_tier: Option<String>,
    pub source: String,
}

#[must_use]
pub fn normalize_provider_usage(facts: ProviderUsageFacts<'_>) -> NormalizedProviderUsage {
    let model = facts
        .response_model
        .unwrap_or(facts.requested_model)
        .to_owned();
    NormalizedProviderUsage {
        requested_model_metadata: (model != facts.requested_model)
            .then(|| facts.requested_model.to_owned()),
        model,
        upstream_provider: facts.upstream_provider.map(str::to_owned),
        service_tier: facts.service_tier.map(str::to_owned),
        source: facts.source.unwrap_or("unknown").to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::{ProviderUsageFacts, normalize_provider_usage};

    #[test]
    fn response_identity_overrides_the_request_and_preserves_provider_details() {
        let actual = normalize_provider_usage(ProviderUsageFacts {
            requested_model: "requested/model",
            response_model: Some("resolved/model"),
            upstream_provider: Some("SyntheticProvider"),
            service_tier: Some("priority"),
            source: Some("openrouter"),
        });
        assert_eq!(actual.model, "resolved/model");
        assert_eq!(
            actual.requested_model_metadata.as_deref(),
            Some("requested/model"),
        );
        assert_eq!(
            actual.upstream_provider.as_deref(),
            Some("SyntheticProvider"),
        );
        assert_eq!(actual.service_tier.as_deref(), Some("priority"));
        assert_eq!(actual.source, "openrouter");
    }

    #[test]
    fn missing_response_identity_uses_requested_model_and_unknown_source() {
        let actual = normalize_provider_usage(ProviderUsageFacts {
            requested_model: "requested/model",
            response_model: None,
            upstream_provider: None,
            service_tier: None,
            source: None,
        });
        assert_eq!(actual.model, "requested/model");
        assert_eq!(actual.requested_model_metadata, None);
        assert_eq!(actual.upstream_provider, None);
        assert_eq!(actual.service_tier, None);
        assert_eq!(actual.source, "unknown");
    }

    #[test]
    fn identical_resolved_model_needs_no_requested_model_metadata() {
        let actual = normalize_provider_usage(ProviderUsageFacts {
            requested_model: "same/model",
            response_model: Some("same/model"),
            upstream_provider: None,
            service_tier: None,
            source: Some("groq"),
        });
        assert_eq!(actual.requested_model_metadata, None);
        assert_eq!(actual.source, "groq");
    }
}
