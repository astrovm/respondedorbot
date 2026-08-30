//! Provider streaming text hold-and-release state machine.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamToolCall {
    pub index: i64,
    #[serde(default)]
    pub id: String,
    #[serde(default = "default_tool_call_type", rename = "type")]
    pub call_type: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamToolCallFragment {
    pub position: i64,
    pub index: Value,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub call_type: Option<String>,
    pub name: Option<String>,
    pub arguments: Option<String>,
}

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

#[must_use]
pub fn accumulate_stream_tool_calls(
    current: Vec<StreamToolCall>,
    fragments: Vec<StreamToolCallFragment>,
) -> Vec<StreamToolCall> {
    let mut calls = current
        .into_iter()
        .map(|call| (call.index, call))
        .collect::<BTreeMap<_, _>>();
    for fragment in fragments {
        let index = stream_fragment_index(&fragment.index).unwrap_or(fragment.position);
        let accumulated = calls.entry(index).or_insert_with(|| StreamToolCall {
            index,
            id: String::new(),
            call_type: default_tool_call_type(),
            name: String::new(),
            arguments: String::new(),
        });
        append_nonempty(&mut accumulated.id, fragment.id);
        if let Some(call_type) = fragment.call_type.filter(|value| !value.is_empty()) {
            accumulated.call_type = call_type;
        }
        append_nonempty(&mut accumulated.name, fragment.name);
        append_nonempty(&mut accumulated.arguments, fragment.arguments);
    }
    calls.into_values().collect()
}

fn stream_fragment_index(value: &Value) -> Option<i64> {
    match value {
        Value::Null => None,
        Value::Bool(value) => Some(i64::from(*value)),
        Value::Number(value) => value
            .as_i64()
            .or_else(|| value.as_u64().and_then(|number| i64::try_from(number).ok()))
            .or_else(|| {
                value.as_f64().and_then(|number| {
                    (number.is_finite() && number >= i64::MIN as f64 && number <= i64::MAX as f64)
                        .then(|| number.trunc() as i64)
                })
            }),
        Value::String(value) => value.trim().parse::<i64>().ok(),
        Value::Array(_) | Value::Object(_) => None,
    }
}

fn append_nonempty(target: &mut String, fragment: Option<String>) {
    if let Some(fragment) = fragment.filter(|value| !value.is_empty()) {
        target.push_str(&fragment);
    }
}

fn default_tool_call_type() -> String {
    "function".to_owned()
}

#[cfg(test)]
mod tests {
    use super::{
        StreamTextDecision, StreamToolCall, StreamToolCallFragment, accumulate_stream_tool_calls,
        apply_stream_text, could_be_pseudo_tool_call,
    };
    use serde_json::json;

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

    #[test]
    fn accumulates_interleaved_tool_call_fragments_in_index_order() {
        let actual = accumulate_stream_tool_calls(
            vec![],
            vec![
                StreamToolCallFragment {
                    position: 0,
                    index: json!(1),
                    id: Some("call_b".to_owned()),
                    call_type: Some("function".to_owned()),
                    name: Some("web_".to_owned()),
                    arguments: Some("{\"url\":".to_owned()),
                },
                StreamToolCallFragment {
                    position: 1,
                    index: json!("0"),
                    id: Some("call_a".to_owned()),
                    call_type: None,
                    name: Some("calc".to_owned()),
                    arguments: Some("{\"x\":1}".to_owned()),
                },
                StreamToolCallFragment {
                    position: 0,
                    index: json!(1),
                    id: None,
                    call_type: None,
                    name: Some("fetch".to_owned()),
                    arguments: Some("\"https://example.com\"}".to_owned()),
                },
            ],
        );
        assert_eq!(
            actual,
            vec![
                StreamToolCall {
                    index: 0,
                    id: "call_a".to_owned(),
                    call_type: "function".to_owned(),
                    name: "calc".to_owned(),
                    arguments: "{\"x\":1}".to_owned(),
                },
                StreamToolCall {
                    index: 1,
                    id: "call_b".to_owned(),
                    call_type: "function".to_owned(),
                    name: "web_fetch".to_owned(),
                    arguments: "{\"url\":\"https://example.com\"}".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn updates_existing_calls_and_falls_back_to_fragment_position() {
        let actual = accumulate_stream_tool_calls(
            vec![StreamToolCall {
                index: -1,
                id: "prefix".to_owned(),
                call_type: "custom".to_owned(),
                name: "tool".to_owned(),
                arguments: "{".to_owned(),
            }],
            vec![
                StreamToolCallFragment {
                    position: -1,
                    index: json!([]),
                    id: Some("_suffix".to_owned()),
                    call_type: Some("function".to_owned()),
                    name: None,
                    arguments: Some("}".to_owned()),
                },
                StreamToolCallFragment {
                    position: 2,
                    index: json!(2.9),
                    id: None,
                    call_type: None,
                    name: Some("other".to_owned()),
                    arguments: None,
                },
            ],
        );
        assert_eq!(actual[0].id, "prefix_suffix");
        assert_eq!(actual[0].call_type, "function");
        assert_eq!(actual[0].arguments, "{}");
        assert_eq!(actual[1].index, 2);
        assert_eq!(actual[1].name, "other");
    }
}
