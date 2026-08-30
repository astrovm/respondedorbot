//! Tool-call argument and registry-availability policy.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ToolArgumentError {
    #[error("invalid tool arguments JSON: {0}")]
    InvalidJson(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolArguments {
    object_json: String,
}

impl ParsedToolArguments {
    #[must_use]
    pub fn object_json(&self) -> &str {
        &self.object_json
    }
}

/// Validate one provider argument string and preserve only JSON objects.
pub fn parse_tool_arguments(raw: &str) -> Result<ParsedToolArguments, ToolArgumentError> {
    let value: Value = serde_json::from_str(raw)
        .map_err(|error| ToolArgumentError::InvalidJson(error.to_string()))?;
    let object_json = if value.is_object() {
        raw.to_owned()
    } else {
        "{}".to_owned()
    };
    Ok(ParsedToolArguments { object_json })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolAvailabilityFacts {
    pub environment_requirements_met: bool,
    pub context_requirements_met: bool,
    pub task_allowed: bool,
}

#[must_use]
pub fn select_available_tools(
    tools: &[ToolAvailabilityFacts],
    context_provided: bool,
    task_mode: bool,
) -> Vec<usize> {
    tools
        .iter()
        .enumerate()
        .filter_map(|(index, tool)| {
            let registry_filtering = context_provided || task_mode;
            let available = (!registry_filtering || tool.environment_requirements_met)
                && (!context_provided || tool.context_requirements_met)
                && (!task_mode || tool.task_allowed);
            available.then_some(index)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        ParsedToolArguments, ToolArgumentError, ToolAvailabilityFacts, parse_tool_arguments,
        select_available_tools,
    };

    #[test]
    fn argument_parser_accepts_objects_and_normalizes_non_objects() {
        assert_eq!(
            parse_tool_arguments(r#"{"query":"rust","limit":3}"#),
            Ok(ParsedToolArguments {
                object_json: r#"{"query":"rust","limit":3}"#.to_owned(),
            }),
        );
        assert_eq!(
            parse_tool_arguments(r#"["not", "an", "object"]"#),
            Ok(ParsedToolArguments {
                object_json: "{}".to_owned(),
            }),
        );
        assert_eq!(
            parse_tool_arguments("null"),
            Ok(ParsedToolArguments {
                object_json: "{}".to_owned(),
            }),
        );
        assert!(matches!(
            parse_tool_arguments("not JSON"),
            Err(ToolArgumentError::InvalidJson(_))
        ));
    }

    #[test]
    fn availability_preserves_environment_context_and_task_rules() {
        let tools = [
            ToolAvailabilityFacts {
                environment_requirements_met: true,
                context_requirements_met: false,
                task_allowed: true,
            },
            ToolAvailabilityFacts {
                environment_requirements_met: false,
                context_requirements_met: true,
                task_allowed: true,
            },
            ToolAvailabilityFacts {
                environment_requirements_met: true,
                context_requirements_met: true,
                task_allowed: false,
            },
        ];

        assert_eq!(select_available_tools(&tools, false, false), vec![0, 1, 2]);
        assert_eq!(select_available_tools(&tools, false, true), vec![0]);
        assert_eq!(select_available_tools(&tools, true, false), vec![2]);
        assert_eq!(
            select_available_tools(&tools, true, true),
            Vec::<usize>::new()
        );
    }
}
