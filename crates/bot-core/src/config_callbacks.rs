//! Pure chat-configuration callback state transitions.

use num_bigint::BigInt;

use crate::chat_config::ChatConfig;
use crate::config_command::{TIMEZONE_OFFSET_MAX, TIMEZONE_OFFSET_MIN};

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

pub enum ConfigCallbackDiagnostic {
    InvalidTimezone,
    InvalidCreditlessLimit,
}

pub enum ConfigCallbackOutcome {
    NotHandled,
    LegacyRequired,
    Guard,
    Render {
        changed: bool,
        diagnostic: Option<ConfigCallbackDiagnostic>,
    },
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

fn parsed_timezone(value: &str) -> Option<i64> {
    let parsed = value.parse::<BigInt>().ok()?;
    let minimum = BigInt::from(TIMEZONE_OFFSET_MIN);
    let maximum = BigInt::from(TIMEZONE_OFFSET_MAX);
    if parsed < minimum {
        Some(TIMEZONE_OFFSET_MIN)
    } else if parsed > maximum {
        Some(TIMEZONE_OFFSET_MAX)
    } else {
        parsed.try_into().ok()
    }
}

/// Plan a complete `cfg:*` transition while preserving Python integer behavior.
#[must_use]
pub fn plan_config_callback(
    data: &str,
    current: &ChatConfig,
) -> (ConfigCallbackOutcome, ChatConfig) {
    let Some(payload) = data.strip_prefix("cfg:") else {
        return (ConfigCallbackOutcome::NotHandled, current.clone());
    };
    let Some((action, value)) = payload.split_once(':') else {
        return (ConfigCallbackOutcome::Guard, current.clone());
    };
    let current_toggle = match action {
        "random" => Some(current.ai_random_replies),
        "followups" => Some(current.ai_command_followups),
        "linkfixfollowups" => Some(current.ignore_link_fix_followups),
        _ => None,
    };
    if action == "creditless"
        && ((value == "increase" && current.creditless_user_hourly_limit == i64::MAX)
            || (!matches!(
                value,
                "none" | "decrease" | "increase" | "unlimited" | "current"
            ) && value.parse::<BigInt>().is_ok()
                && value.parse::<i64>().is_err()))
    {
        return (ConfigCallbackOutcome::LegacyRequired, current.clone());
    }
    let numeric_value = if action == "timezone" {
        parsed_timezone(value)
    } else {
        value.parse().ok()
    };
    let evaluation = evaluate_config_callback(&ConfigCallbackInput {
        action: action.to_owned(),
        value: value.to_owned(),
        current_toggle,
        current_creditless_limit: (action == "creditless")
            .then_some(current.creditless_user_hourly_limit),
        numeric_value,
        timezone_min: TIMEZONE_OFFSET_MIN,
        timezone_max: TIMEZONE_OFFSET_MAX,
    });
    let mut config = current.clone();
    let mut changed = true;
    let diagnostic = match evaluation {
        ConfigCallbackEvaluation::NoChange => {
            changed = false;
            None
        }
        ConfigCallbackEvaluation::GuardCurrent => {
            return (ConfigCallbackOutcome::Guard, current.clone());
        }
        ConfigCallbackEvaluation::InvalidTimezone => {
            changed = false;
            Some(ConfigCallbackDiagnostic::InvalidTimezone)
        }
        ConfigCallbackEvaluation::InvalidCreditlessLimit => {
            changed = false;
            Some(ConfigCallbackDiagnostic::InvalidCreditlessLimit)
        }
        ConfigCallbackEvaluation::SetLanguage(value) => {
            config.language = value;
            None
        }
        ConfigCallbackEvaluation::SetLinkMode(value) => {
            config.link_mode = value;
            None
        }
        ConfigCallbackEvaluation::SetToggle { field, value } => {
            match field {
                ToggleField::RandomReplies => config.ai_random_replies = value,
                ToggleField::CommandFollowups => config.ai_command_followups = value,
                ToggleField::IgnoreLinkFixFollowups => {
                    config.ignore_link_fix_followups = value;
                }
            }
            None
        }
        ConfigCallbackEvaluation::SetTimezone(value) => {
            config.timezone_offset = value;
            None
        }
        ConfigCallbackEvaluation::SetCreditlessLimit(value) => {
            config.creditless_user_hourly_limit = value;
            None
        }
    };
    (
        ConfigCallbackOutcome::Render {
            changed,
            diagnostic,
        },
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        ConfigCallbackDiagnostic, ConfigCallbackEvaluation, ConfigCallbackInput,
        ConfigCallbackOutcome, ToggleField, evaluate_config_callback, plan_config_callback,
    };
    use crate::chat_config::ChatConfig;
    use crate::config_command::TIMEZONE_OFFSET_MAX;

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

    fn diagnostic_name(diagnostic: Option<ConfigCallbackDiagnostic>) -> Option<&'static str> {
        diagnostic.map(|diagnostic| match diagnostic {
            ConfigCallbackDiagnostic::InvalidTimezone => "timezone",
            ConfigCallbackDiagnostic::InvalidCreditlessLimit => "creditless",
        })
    }

    fn render_matches(
        plan: (ConfigCallbackOutcome, ChatConfig),
        expected: &ChatConfig,
        expected_changed: bool,
        expected_diagnostic: Option<&str>,
    ) -> bool {
        match plan {
            (
                ConfigCallbackOutcome::Render {
                    changed,
                    diagnostic,
                },
                config,
            ) => {
                config == *expected
                    && changed == expected_changed
                    && diagnostic_name(diagnostic) == expected_diagnostic
            }
            (
                ConfigCallbackOutcome::NotHandled
                | ConfigCallbackOutcome::LegacyRequired
                | ConfigCallbackOutcome::Guard,
                _,
            ) => false,
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

    #[test]
    fn complete_plan_mutates_typed_config_and_preserves_guard_behavior() {
        let current = ChatConfig::default();
        assert!(render_matches(
            plan_config_callback("cfg:random:toggle", &current),
            &ChatConfig {
                ai_random_replies: false,
                ..current.clone()
            },
            true,
            None,
        ));
        assert!(matches!(
            plan_config_callback("cfg:timezone:current", &current),
            (ConfigCallbackOutcome::Guard, _)
        ));
        assert!(matches!(
            plan_config_callback("cfg:broken", &current),
            (ConfigCallbackOutcome::Guard, _)
        ));
        assert!(matches!(
            plan_config_callback("other", &current),
            (ConfigCallbackOutcome::NotHandled, _)
        ));
    }

    #[test]
    fn complete_plan_clamps_arbitrary_timezone_and_defers_unrepresentable_limits() {
        let current = ChatConfig::default();
        assert!(render_matches(
            plan_config_callback(
                "cfg:timezone:999999999999999999999999999999999999",
                &current,
            ),
            &ChatConfig {
                timezone_offset: TIMEZONE_OFFSET_MAX,
                ..current.clone()
            },
            true,
            None,
        ));
        assert!(matches!(
            plan_config_callback(
                "cfg:creditless:999999999999999999999999999999999999",
                &current,
            ),
            (ConfigCallbackOutcome::LegacyRequired, _)
        ));
    }

    #[test]
    fn complete_plan_reports_invalid_values_but_still_renders_current_config() {
        let current = ChatConfig::default();
        assert!(render_matches(
            plan_config_callback("cfg:timezone:later", &current),
            &current,
            false,
            Some("timezone"),
        ));
        assert!(render_matches(
            plan_config_callback("cfg:creditless:-2", &current),
            &current,
            false,
            Some("creditless"),
        ));
    }

    #[test]
    fn complete_plan_applies_every_supported_field_transition() {
        let current = ChatConfig::default();
        for (data, expected) in [
            (
                "cfg:language:en",
                ChatConfig {
                    language: "en".to_owned(),
                    ..current.clone()
                },
            ),
            (
                "cfg:link:delete",
                ChatConfig {
                    link_mode: "delete".to_owned(),
                    ..current.clone()
                },
            ),
            (
                "cfg:followups:any-value",
                ChatConfig {
                    ai_command_followups: false,
                    ..current.clone()
                },
            ),
            (
                "cfg:linkfixfollowups:toggle",
                ChatConfig {
                    ignore_link_fix_followups: false,
                    ..current.clone()
                },
            ),
            (
                "cfg:timezone:-999999999999999999999999999999999999",
                ChatConfig {
                    timezone_offset: super::TIMEZONE_OFFSET_MIN,
                    ..current.clone()
                },
            ),
            (
                "cfg:timezone:2",
                ChatConfig {
                    timezone_offset: 2,
                    ..current.clone()
                },
            ),
            (
                "cfg:creditless:unlimited",
                ChatConfig {
                    creditless_user_hourly_limit: -1,
                    ..current.clone()
                },
            ),
        ] {
            assert!(render_matches(
                plan_config_callback(data, &current),
                &expected,
                true,
                None,
            ));
        }

        let maximum = ChatConfig {
            creditless_user_hourly_limit: i64::MAX,
            ..current
        };
        assert!(matches!(
            plan_config_callback("cfg:creditless:increase", &maximum),
            (ConfigCallbackOutcome::LegacyRequired, _)
        ));
    }

    #[test]
    fn derived_traits_cover_every_callback_outcome_shape() {
        let input = input("random", "toggle");
        assert_eq!(input.clone(), input);
        assert!(!format!("{input:?}").is_empty());

        let evaluations = [
            ConfigCallbackEvaluation::NoChange,
            ConfigCallbackEvaluation::GuardCurrent,
            ConfigCallbackEvaluation::InvalidTimezone,
            ConfigCallbackEvaluation::InvalidCreditlessLimit,
            ConfigCallbackEvaluation::SetLanguage("en".to_owned()),
            ConfigCallbackEvaluation::SetLinkMode("off".to_owned()),
            ConfigCallbackEvaluation::SetToggle {
                field: ToggleField::CommandFollowups,
                value: false,
            },
            ConfigCallbackEvaluation::SetTimezone(2),
            ConfigCallbackEvaluation::SetCreditlessLimit(7),
        ];
        for evaluation in evaluations {
            assert_eq!(evaluation.clone(), evaluation);
            assert!(!format!("{evaluation:?}").is_empty());
        }
    }
}
