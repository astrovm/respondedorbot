//! Provider failure classification for retry and fallback policy.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderErrorFacts<'a> {
    pub status_code: Option<i64>,
    pub status: Option<i64>,
    pub code: &'a str,
    pub message: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderErrorPolicy {
    pub rate_limited: bool,
    pub try_next_groq_account: bool,
}

#[must_use]
pub fn classify_provider_error(facts: ProviderErrorFacts<'_>) -> ProviderErrorPolicy {
    let message = facts.message.to_lowercase();
    ProviderErrorPolicy {
        rate_limited: facts.status_code == Some(429)
            || facts.status == Some(429)
            || message.contains("rate limit")
            || message.contains("429"),
        try_next_groq_account: facts.status_code == Some(413)
            || facts.status == Some(413)
            || facts.code.trim().eq_ignore_ascii_case("request_too_large")
            || message.contains("request_too_large")
            || message.contains("payload too large"),
    }
}

#[cfg(test)]
mod tests {
    use super::{ProviderErrorFacts, ProviderErrorPolicy, classify_provider_error};

    #[test]
    fn classifies_status_code_status_code_string_and_message_variants() {
        for (facts, expected) in [
            (
                ProviderErrorFacts {
                    status_code: Some(429),
                    status: None,
                    code: "",
                    message: "unrelated",
                },
                ProviderErrorPolicy {
                    rate_limited: true,
                    try_next_groq_account: false,
                },
            ),
            (
                ProviderErrorFacts {
                    status_code: None,
                    status: Some(413),
                    code: "",
                    message: "unrelated",
                },
                ProviderErrorPolicy {
                    rate_limited: false,
                    try_next_groq_account: true,
                },
            ),
            (
                ProviderErrorFacts {
                    status_code: None,
                    status: None,
                    code: " REQUEST_TOO_LARGE ",
                    message: "unrelated",
                },
                ProviderErrorPolicy {
                    rate_limited: false,
                    try_next_groq_account: true,
                },
            ),
            (
                ProviderErrorFacts {
                    status_code: None,
                    status: None,
                    code: "",
                    message: "Error 429: RATE LIMIT reached; payload too large",
                },
                ProviderErrorPolicy {
                    rate_limited: true,
                    try_next_groq_account: true,
                },
            ),
            (
                ProviderErrorFacts {
                    status_code: Some(500),
                    status: None,
                    code: "server_error",
                    message: "try again",
                },
                ProviderErrorPolicy {
                    rate_limited: false,
                    try_next_groq_account: false,
                },
            ),
        ] {
            assert_eq!(classify_provider_error(facts), expected);
        }
    }
}
