//! Provider-chain ordering and completion outcome policy.

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderChainOutcome {
    pub provider_name: String,
    pub fallback_used: bool,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProviderChainError {
    #[error(
        "successful provider position {position} is outside {provider_count} available providers"
    )]
    InvalidSuccessPosition {
        position: usize,
        provider_count: usize,
    },
}

#[must_use]
pub fn available_provider_indices(availability: &[bool]) -> Vec<usize> {
    availability
        .iter()
        .enumerate()
        .filter_map(|(index, available)| available.then_some(index))
        .collect()
}

pub fn completion_outcome(
    available_provider_names: &[String],
    successful_position: Option<usize>,
) -> Result<ProviderChainOutcome, ProviderChainError> {
    if let Some(position) = successful_position {
        let provider_name = available_provider_names.get(position).ok_or(
            ProviderChainError::InvalidSuccessPosition {
                position,
                provider_count: available_provider_names.len(),
            },
        )?;
        return Ok(ProviderChainOutcome {
            provider_name: provider_name.clone(),
            fallback_used: position > 0,
        });
    }

    Ok(ProviderChainOutcome {
        provider_name: available_provider_names
            .last()
            .cloned()
            .unwrap_or_else(|| "none".to_owned()),
        fallback_used: false,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ProviderChainError, ProviderChainOutcome, available_provider_indices, completion_outcome,
    };

    #[test]
    fn selection_keeps_configured_order() {
        assert_eq!(
            available_provider_indices(&[false, true, true, false]),
            vec![1, 2],
        );
        assert!(available_provider_indices(&[]).is_empty());
    }

    #[test]
    fn outcome_tracks_fallback_and_last_attempt() {
        let names = vec!["primary".to_owned(), "fallback".to_owned()];
        assert_eq!(
            completion_outcome(&names, Some(0)),
            Ok(ProviderChainOutcome {
                provider_name: "primary".to_owned(),
                fallback_used: false,
            }),
        );
        assert_eq!(
            completion_outcome(&names, Some(1)),
            Ok(ProviderChainOutcome {
                provider_name: "fallback".to_owned(),
                fallback_used: true,
            }),
        );
        assert_eq!(
            completion_outcome(&names, None),
            Ok(ProviderChainOutcome {
                provider_name: "fallback".to_owned(),
                fallback_used: false,
            }),
        );
        assert_eq!(
            completion_outcome(&[], None),
            Ok(ProviderChainOutcome {
                provider_name: "none".to_owned(),
                fallback_used: false,
            }),
        );
        assert_eq!(
            completion_outcome(&names, Some(2)),
            Err(ProviderChainError::InvalidSuccessPosition {
                position: 2,
                provider_count: 2,
            }),
        );
    }
}
