//! Pure chat-configuration callback state transitions.

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfigCallbackInput {
    pub action: String,
    pub value: String,
    pub current_toggle: Option<bool>,
    pub current_creditless_limit: Option<i64>,
    pub numeric_value: Option<i64>,
    pub timezone_min: i64,
    pub timezone_max: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToggleField {
    RandomReplies,
    CommandFollowups,
    IgnoreLinkFixFollowups,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConfigCallbackEvaluation {
    NoChange,
    GuardCurrent,
    InvalidTimezone,
    InvalidCreditlessLimit,
    SetLanguage(String),
    SetLinkMode(String),
    SetToggle { field: ToggleField, value: bool },
    SetTimezone(i64),
    SetCreditlessLimit(i64),
}

fn toggle_field(action: &str) -> Option<ToggleField> {
    match action {
        "random" => Some(ToggleField::RandomReplies),
        "followups" => Some(ToggleField::CommandFollowups),
        "linkfixfollowups" => Some(ToggleField::IgnoreLinkFixFollowups),
        _ => None,
    }
}

fn evaluate_creditless(input: &ConfigCallbackInput) -> ConfigCallbackEvaluation {
    if input.value == "current" {
        return ConfigCallbackEvaluation::GuardCurrent;
    }
    let current = input.current_creditless_limit;
    let limit = match input.value.as_str() {
        "none" => Some(0),
        "decrease" => current.map(|value| if value < 0 { value } else { (value - 1).max(0) }),
        "increase" => current.map(|value| {
            if value < 0 {
                value
            } else {
                value.saturating_add(1)
            }
        }),
        "unlimited" => Some(-1),
        _ => input.numeric_value.filter(|value| *value >= -1),
    };
    limit.map_or(
        ConfigCallbackEvaluation::InvalidCreditlessLimit,
        ConfigCallbackEvaluation::SetCreditlessLimit,
    )
}

/// Evaluate one `cfg:*` callback without executing persistence or Telegram I/O.
#[must_use]
pub fn evaluate_config_callback(input: &ConfigCallbackInput) -> ConfigCallbackEvaluation {
    match (input.action.as_str(), input.value.as_str()) {
        ("language", value @ ("es" | "en")) => {
            ConfigCallbackEvaluation::SetLanguage(value.to_owned())
        }
        ("link", value @ ("reply" | "delete" | "off")) => {
            ConfigCallbackEvaluation::SetLinkMode(value.to_owned())
        }
        ("timezone", "current") => ConfigCallbackEvaluation::GuardCurrent,
        ("timezone", _) => {
            input
                .numeric_value
                .map_or(ConfigCallbackEvaluation::InvalidTimezone, |value| {
                    ConfigCallbackEvaluation::SetTimezone(
                        value.clamp(input.timezone_min, input.timezone_max),
                    )
                })
        }
        ("creditless", _) => evaluate_creditless(input),
        (action, _) => toggle_field(action).zip(input.current_toggle).map_or(
            ConfigCallbackEvaluation::NoChange,
            |(field, current)| ConfigCallbackEvaluation::SetToggle {
                field,
                value: !current,
            },
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ConfigCallbackEvaluation, ConfigCallbackInput, ToggleField, evaluate_config_callback,
    };

    fn input(action: &str, value: &str) -> ConfigCallbackInput {
        ConfigCallbackInput {
            action: action.to_owned(),
            value: value.to_owned(),
            current_toggle: Some(true),
            current_creditless_limit: Some(5),
            numeric_value: None,
            timezone_min: -12,
            timezone_max: 14,
        }
    }

    #[test]
    fn selects_language_link_and_toggle_updates() {
        assert_eq!(
            evaluate_config_callback(&input("language", "en")),
            ConfigCallbackEvaluation::SetLanguage("en".to_owned())
        );
        assert_eq!(
            evaluate_config_callback(&input("link", "delete")),
            ConfigCallbackEvaluation::SetLinkMode("delete".to_owned())
        );
        assert_eq!(
            evaluate_config_callback(&input("random", "toggle")),
            ConfigCallbackEvaluation::SetToggle {
                field: ToggleField::RandomReplies,
                value: false,
            }
        );
        let mut missing_current = input("followups", "toggle");
        missing_current.current_toggle = None;
        assert_eq!(
            evaluate_config_callback(&missing_current),
            ConfigCallbackEvaluation::NoChange
        );
    }

    #[test]
    fn clamps_timezones_and_distinguishes_current_from_invalid() {
        let mut timezone = input("timezone", "99");
        timezone.numeric_value = Some(99);
        assert_eq!(
            evaluate_config_callback(&timezone),
            ConfigCallbackEvaluation::SetTimezone(14)
        );
        assert_eq!(
            evaluate_config_callback(&input("timezone", "current")),
            ConfigCallbackEvaluation::GuardCurrent
        );
        assert_eq!(
            evaluate_config_callback(&input("timezone", "invalid")),
            ConfigCallbackEvaluation::InvalidTimezone
        );
    }

    #[test]
    fn evaluates_every_creditless_stepper_operation() {
        for (value, expected) in [
            ("none", ConfigCallbackEvaluation::SetCreditlessLimit(0)),
            ("decrease", ConfigCallbackEvaluation::SetCreditlessLimit(4)),
            ("increase", ConfigCallbackEvaluation::SetCreditlessLimit(6)),
            (
                "unlimited",
                ConfigCallbackEvaluation::SetCreditlessLimit(-1),
            ),
            ("current", ConfigCallbackEvaluation::GuardCurrent),
            ("invalid", ConfigCallbackEvaluation::InvalidCreditlessLimit),
        ] {
            assert_eq!(
                evaluate_config_callback(&input("creditless", value)),
                expected
            );
        }

        let mut unlimited = input("creditless", "increase");
        unlimited.current_creditless_limit = Some(-1);
        assert_eq!(
            evaluate_config_callback(&unlimited),
            ConfigCallbackEvaluation::SetCreditlessLimit(-1)
        );
        let mut explicit = input("creditless", "12");
        explicit.numeric_value = Some(12);
        assert_eq!(
            evaluate_config_callback(&explicit),
            ConfigCallbackEvaluation::SetCreditlessLimit(12)
        );
        explicit.numeric_value = Some(-2);
        assert_eq!(
            evaluate_config_callback(&explicit),
            ConfigCallbackEvaluation::InvalidCreditlessLimit
        );
    }

    #[test]
    fn unknown_or_invalid_choices_do_not_mutate_config() {
        assert_eq!(
            evaluate_config_callback(&input("language", "fr")),
            ConfigCallbackEvaluation::NoChange
        );
        assert_eq!(
            evaluate_config_callback(&input("unknown", "value")),
            ConfigCallbackEvaluation::NoChange
        );
    }
}
