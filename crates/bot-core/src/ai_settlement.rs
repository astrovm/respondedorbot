//! AI orchestration settlement decisions, separate from financial side effects.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettlementAction {
    Noop,
    SettleSuccess,
    SettleUsageBeforeFallback,
    SettleUsageBeforeDeliveryFailure,
    RefundFallback,
    RefundProviderUnavailable,
    RefundDeliveryFailure,
    Continue,
}

impl SettlementAction {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Noop => "noop",
            Self::SettleSuccess => "settle_success",
            Self::SettleUsageBeforeFallback => "settle_usage_before_fallback",
            Self::SettleUsageBeforeDeliveryFailure => "settle_usage_before_delivery_failure",
            Self::RefundFallback => "refund_fallback",
            Self::RefundProviderUnavailable => "refund_provider_unavailable",
            Self::RefundDeliveryFailure => "refund_delivery_failure",
            Self::Continue => "continue",
        }
    }
}

#[must_use]
pub const fn media_settlement_action(
    has_reservation: bool,
    has_billing_segments: bool,
) -> SettlementAction {
    if !has_reservation {
        SettlementAction::Noop
    } else if has_billing_segments {
        SettlementAction::SettleSuccess
    } else {
        SettlementAction::RefundFallback
    }
}

#[must_use]
pub const fn conversation_settlement_action(
    is_fallback: bool,
    has_billing_segments: bool,
) -> SettlementAction {
    if !is_fallback {
        SettlementAction::SettleSuccess
    } else if has_billing_segments {
        SettlementAction::SettleUsageBeforeFallback
    } else {
        SettlementAction::RefundFallback
    }
}

#[must_use]
pub const fn summary_settlement_action(
    provider_unavailable: bool,
    is_fallback: bool,
    has_billing_segments: bool,
) -> SettlementAction {
    if provider_unavailable {
        SettlementAction::RefundProviderUnavailable
    } else if !is_fallback {
        SettlementAction::Continue
    } else if has_billing_segments {
        SettlementAction::SettleUsageBeforeFallback
    } else {
        SettlementAction::RefundFallback
    }
}

#[must_use]
pub const fn delivery_failure_settlement_action(has_billing_segments: bool) -> SettlementAction {
    if has_billing_segments {
        SettlementAction::SettleUsageBeforeDeliveryFailure
    } else {
        SettlementAction::RefundDeliveryFailure
    }
}

#[cfg(test)]
mod tests {
    use super::{
        SettlementAction, conversation_settlement_action, delivery_failure_settlement_action,
        media_settlement_action, summary_settlement_action,
    };

    #[test]
    fn media_settlement_requires_a_reservation_and_preserves_incurred_usage() {
        assert_eq!(
            media_settlement_action(false, false),
            SettlementAction::Noop,
        );
        assert_eq!(media_settlement_action(false, true), SettlementAction::Noop,);
        assert_eq!(
            media_settlement_action(true, true),
            SettlementAction::SettleSuccess,
        );
        assert_eq!(
            media_settlement_action(true, false),
            SettlementAction::RefundFallback,
        );
    }

    #[test]
    fn conversation_fallback_settles_only_usage_that_was_already_incurred() {
        assert_eq!(
            conversation_settlement_action(false, false),
            SettlementAction::SettleSuccess,
        );
        assert_eq!(
            conversation_settlement_action(false, true),
            SettlementAction::SettleSuccess,
        );
        assert_eq!(
            conversation_settlement_action(true, true),
            SettlementAction::SettleUsageBeforeFallback,
        );
        assert_eq!(
            conversation_settlement_action(true, false),
            SettlementAction::RefundFallback,
        );
    }

    #[test]
    fn summary_provider_unavailability_precedes_fallback_and_usage() {
        assert_eq!(
            summary_settlement_action(true, true, true),
            SettlementAction::RefundProviderUnavailable,
        );
        assert_eq!(
            summary_settlement_action(false, false, true),
            SettlementAction::Continue,
        );
        assert_eq!(
            summary_settlement_action(false, true, true),
            SettlementAction::SettleUsageBeforeFallback,
        );
        assert_eq!(
            summary_settlement_action(false, true, false),
            SettlementAction::RefundFallback,
        );
    }

    #[test]
    fn delivery_failure_settles_reported_usage_and_refunds_empty_calls() {
        assert_eq!(
            delivery_failure_settlement_action(true),
            SettlementAction::SettleUsageBeforeDeliveryFailure,
        );
        assert_eq!(
            delivery_failure_settlement_action(false),
            SettlementAction::RefundDeliveryFailure,
        );
    }

    #[test]
    fn serialized_action_names_are_stable() {
        for (action, expected) in [
            (SettlementAction::Noop, "noop"),
            (SettlementAction::SettleSuccess, "settle_success"),
            (
                SettlementAction::SettleUsageBeforeFallback,
                "settle_usage_before_fallback",
            ),
            (
                SettlementAction::SettleUsageBeforeDeliveryFailure,
                "settle_usage_before_delivery_failure",
            ),
            (SettlementAction::RefundFallback, "refund_fallback"),
            (
                SettlementAction::RefundProviderUnavailable,
                "refund_provider_unavailable",
            ),
            (
                SettlementAction::RefundDeliveryFailure,
                "refund_delivery_failure",
            ),
            (SettlementAction::Continue, "continue"),
        ] {
            assert_eq!(action.as_str(), expected);
        }
    }
}
