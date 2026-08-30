//! Side-effect-free classification of provider tool calls.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallAction {
    SkipMissingFunction,
    SkipUnregistered,
    Execute,
}

impl ToolCallAction {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SkipMissingFunction => "skip_missing_function",
            Self::SkipUnregistered => "skip_unregistered",
            Self::Execute => "execute",
        }
    }
}

#[must_use]
pub const fn evaluate_tool_call(has_function: bool, registered: bool) -> ToolCallAction {
    if !has_function {
        ToolCallAction::SkipMissingFunction
    } else if !registered {
        ToolCallAction::SkipUnregistered
    } else {
        ToolCallAction::Execute
    }
}

#[cfg(test)]
mod tests {
    use super::{ToolCallAction, evaluate_tool_call};

    #[test]
    fn classification_requires_a_function_and_registration() {
        assert_eq!(
            evaluate_tool_call(false, false),
            ToolCallAction::SkipMissingFunction,
        );
        assert_eq!(
            evaluate_tool_call(false, true),
            ToolCallAction::SkipMissingFunction,
        );
        assert_eq!(
            evaluate_tool_call(true, false),
            ToolCallAction::SkipUnregistered,
        );
        assert_eq!(evaluate_tool_call(true, true), ToolCallAction::Execute);
    }

    #[test]
    fn action_names_are_stable_for_the_python_bridge() {
        assert_eq!(
            ToolCallAction::SkipMissingFunction.as_str(),
            "skip_missing_function",
        );
        assert_eq!(
            ToolCallAction::SkipUnregistered.as_str(),
            "skip_unregistered",
        );
        assert_eq!(ToolCallAction::Execute.as_str(), "execute");
    }
}
