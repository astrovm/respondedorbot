//! Provider streaming text hold-and-release state machine.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamTextDecision {
    pub held_text: String,
    pub emitted_text: String,
    pub text_released: bool,
}

#[must_use]
pub fn apply_stream_text(
    held_text: &str,
    content: &str,
    hold_all_text: bool,
    text_released: bool,
    possible_pseudo_tools: &[String],
) -> StreamTextDecision {
    if content.is_empty() {
        return StreamTextDecision {
            held_text: held_text.to_owned(),
            emitted_text: String::new(),
            text_released,
        };
    }
    if hold_all_text {
        return StreamTextDecision {
            held_text: format!("{held_text}{content}"),
            emitted_text: String::new(),
            text_released,
        };
    }
    if text_released {
        return StreamTextDecision {
            held_text: held_text.to_owned(),
            emitted_text: content.to_owned(),
            text_released: true,
        };
    }

    let candidate = format!("{held_text}{content}");
    if could_be_pseudo_tool_call(&candidate, possible_pseudo_tools) {
        StreamTextDecision {
            held_text: candidate,
            emitted_text: String::new(),
            text_released: false,
        }
    } else {
        StreamTextDecision {
            held_text: String::new(),
            emitted_text: candidate,
            text_released: true,
        }
    }
}

#[must_use]
pub fn could_be_pseudo_tool_call(text: &str, tool_names: &[String]) -> bool {
    let stripped = text.trim_start();
    if stripped.is_empty() {
        return !tool_names.is_empty();
    }
    tool_names
        .iter()
        .any(|name| name.starts_with(stripped) || stripped.starts_with(&format!("{name}(")))
}

#[cfg(test)]
mod tests {
    use super::{StreamTextDecision, apply_stream_text, could_be_pseudo_tool_call};

    fn tools() -> Vec<String> {
        vec!["web_fetch".to_owned(), "calculate".to_owned()]
    }

    #[test]
    fn holds_partial_pseudo_calls_then_keeps_the_complete_call_private() {
        let first = apply_stream_text("", "web_", false, false, &tools());
        assert_eq!(
            first,
            StreamTextDecision {
                held_text: "web_".to_owned(),
                emitted_text: String::new(),
                text_released: false,
            }
        );
        let second = apply_stream_text(
            &first.held_text,
            "fetch(\"https://example.com\")",
            false,
            first.text_released,
            &tools(),
        );
        assert_eq!(second.held_text, "web_fetch(\"https://example.com\")");
        assert!(second.emitted_text.is_empty());
        assert!(!second.text_released);
    }

    #[test]
    fn releases_plain_text_once_then_streams_subsequent_chunks_directly() {
        let first = apply_stream_text("", "ordinary", false, false, &tools());
        assert_eq!(first.held_text, "");
        assert_eq!(first.emitted_text, "ordinary");
        assert!(first.text_released);
        assert_eq!(
            apply_stream_text("", " text", false, true, &tools()),
            StreamTextDecision {
                held_text: String::new(),
                emitted_text: " text".to_owned(),
                text_released: true,
            }
        );
    }

    #[test]
    fn hold_all_accumulates_without_changing_release_state() {
        assert_eq!(
            apply_stream_text("prior", " answer", true, false, &tools()),
            StreamTextDecision {
                held_text: "prior answer".to_owned(),
                emitted_text: String::new(),
                text_released: false,
            }
        );
        assert_eq!(
            apply_stream_text("prior", "", false, true, &tools()),
            StreamTextDecision {
                held_text: "prior".to_owned(),
                emitted_text: String::new(),
                text_released: true,
            }
        );
    }

    #[test]
    fn candidate_detection_preserves_prefix_and_whitespace_rules() {
        assert!(could_be_pseudo_tool_call("", &tools()));
        assert!(could_be_pseudo_tool_call("  web_f", &tools()));
        assert!(could_be_pseudo_tool_call("web_fetch(", &tools()));
        assert!(!could_be_pseudo_tool_call("web", &[]));
        assert!(!could_be_pseudo_tool_call("plain answer", &tools()));
    }
}
