//! One argument check for every native tool, driven by its provider schema.
//!
//! Models sometimes send `"true"` for a boolean, `10800.0` for an integer, or
//! a field the tool does not have. Reading those loosely turned a wrong type
//! into a silent default (an `anonymous: "true"` poll became public) and an
//! unknown field into nothing at all. This check accepts the obvious forms
//! and rejects the rest by name, so the model can fix the call.

use serde_json::{Map, Value};

use crate::tool_requests::json_integer;

/// Normalize `arguments` against a tool's `parameters` schema.
///
/// Only top-level fields are checked; nested objects keep their own,
/// more specific validation in the tool.
pub fn check_arguments(name: &str, parameters: &Value, arguments: Value) -> Result<Value, String> {
    let Value::Object(arguments) = arguments else {
        return Ok(arguments);
    };
    let Some(properties) = parameters.get("properties").and_then(Value::as_object) else {
        return Ok(Value::Object(arguments));
    };
    let mut checked = Map::new();
    let mut problems = Vec::new();
    for (field, value) in arguments {
        // Some models send every schema field and leave the unused ones null.
        if value.is_null() {
            continue;
        }
        let Some(property) = properties.get(&field) else {
            problems.push(format!(
                "'{field}' is not a parameter (use: {})",
                properties.keys().cloned().collect::<Vec<_>>().join(", ")
            ));
            continue;
        };
        match check_value(property, value) {
            Ok(value) => {
                checked.insert(field, value);
            }
            Err(expected) => problems.push(format!("'{field}' must be {expected}")),
        }
    }
    if problems.is_empty() {
        Ok(Value::Object(checked))
    } else {
        Err(format!(
            "Invalid arguments for '{name}': {}. Call it again with arguments that match its schema.",
            problems.join("; ")
        ))
    }
}

fn check_value(property: &Value, value: Value) -> Result<Value, String> {
    match property.get("type").and_then(Value::as_str) {
        Some("string") => check_string(property, value),
        Some("integer") => json_integer(&value)
            .map(Value::from)
            .ok_or_else(|| "a whole number".to_owned()),
        Some("boolean") => check_boolean(value),
        Some("array") => check_array(value),
        Some("object") => check_object(value),
        _ => Ok(value),
    }
}

fn scalar_text(value: Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(flag) => Some(flag.to_string()),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

fn check_string(property: &Value, value: Value) -> Result<Value, String> {
    let text = scalar_text(value).ok_or_else(|| "text".to_owned())?;
    let Some(allowed) = property.get("enum").and_then(Value::as_array) else {
        return Ok(Value::String(text));
    };
    let wanted = text.trim();
    allowed
        .iter()
        .filter_map(Value::as_str)
        .find(|option| option.eq_ignore_ascii_case(wanted))
        .map(|option| Value::String(option.to_owned()))
        .ok_or_else(|| {
            format!(
                "one of {}",
                allowed
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

fn check_boolean(value: Value) -> Result<Value, String> {
    let flag = match &value {
        Value::Bool(flag) => Some(*flag),
        Value::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        },
        _ => None,
    };
    flag.map(Value::Bool)
        .ok_or_else(|| "true or false".to_owned())
}

fn check_array(value: Value) -> Result<Value, String> {
    let items = match value {
        Value::Array(items) => items,
        // One value where a list belongs is a list of one.
        Value::String(_) | Value::Number(_) | Value::Bool(_) => vec![value],
        Value::Null | Value::Object(_) => return Err("a list of text values".to_owned()),
    };
    items
        .into_iter()
        .map(|item| scalar_text(item).map(Value::String))
        .collect::<Option<Vec<_>>>()
        .map(Value::Array)
        .ok_or_else(|| "a list of text values".to_owned())
}

fn check_object(value: Value) -> Result<Value, String> {
    match value {
        Value::Object(_) => Ok(value),
        // Some models send a nested object as a JSON string.
        Value::String(text) => match serde_json::from_str::<Value>(&text) {
            Ok(parsed @ Value::Object(_)) => Ok(parsed),
            _ => Err("an object".to_owned()),
        },
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::Array(_) => {
            Err("an object".to_owned())
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::check_arguments;

    fn schema() -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
                "anonymous": {"type": "boolean"},
                "limit": {"type": "integer"},
                "timeframe": {"type": "string", "enum": ["1h", "24h"]},
                "config": {"type": "object"},
                "loose": {"description": "no declared type"}
            }
        })
    }

    #[test]
    fn obvious_forms_are_normalized() {
        assert_eq!(
            check_arguments(
                "synthetic_tool",
                &schema(),
                json!({
                    "question": 42,
                    "options": ["option-alpha", 2, true],
                    "anonymous": " TRUE ",
                    "limit": "7",
                    "timeframe": "24H",
                    "config": "{\"type\":\"cron\"}",
                    "loose": [1],
                    "unused": null
                }),
            ),
            Ok(json!({
                "question": "42",
                "options": ["option-alpha", "2", "true"],
                "anonymous": true,
                "limit": 7,
                "timeframe": "24h",
                "config": {"type": "cron"},
                "loose": [1]
            }))
        );
        assert_eq!(
            check_arguments(
                "synthetic_tool",
                &schema(),
                json!({"options": "option-alpha", "anonymous": false, "limit": 3.0, "config": {"a": 1}}),
            ),
            Ok(
                json!({"options": ["option-alpha"], "anonymous": false, "limit": 3, "config": {"a": 1}})
            )
        );
        assert_eq!(
            check_arguments("synthetic_tool", &schema(), json!({"anonymous": "false"})),
            Ok(json!({"anonymous": false}))
        );
    }

    #[test]
    fn broken_and_unknown_fields_are_rejected_by_name() {
        let error = check_arguments(
            "synthetic_tool",
            &schema(),
            json!({
                "question": ["not", "text"],
                "options": [{"nested": true}],
                "anonymous": "yes",
                "limit": "3h",
                "timeframe": "2h",
                "config": "not json",
                "every": "3h"
            }),
        )
        .err()
        .unwrap_or_default();
        for expected in [
            "Invalid arguments for 'synthetic_tool': ",
            "'question' must be text",
            "'options' must be a list of text values",
            "'anonymous' must be true or false",
            "'limit' must be a whole number",
            "'timeframe' must be one of 1h, 24h",
            "'config' must be an object",
            "'every' is not a parameter (use: ",
            "Call it again with arguments that match its schema.",
        ] {
            assert!(error.contains(expected), "{expected} missing from {error}");
        }
        for (arguments, expected) in [
            (
                json!({"options": {"a": 1}}),
                "'options' must be a list of text values",
            ),
            (json!({"anonymous": 1}), "'anonymous' must be true or false"),
            (json!({"config": "[1]"}), "'config' must be an object"),
            (json!({"config": 5}), "'config' must be an object"),
            (json!({"question": {"a": 1}}), "'question' must be text"),
        ] {
            let result = check_arguments("synthetic_tool", &schema(), arguments);
            assert!(
                result.as_ref().is_err_and(|error| error.contains(expected)),
                "{result:?}"
            );
        }
    }

    #[test]
    fn schemas_without_properties_and_non_objects_pass_through() {
        assert_eq!(
            check_arguments("synthetic_tool", &json!({}), json!({"anything": 1})),
            Ok(json!({"anything": 1}))
        );
        assert_eq!(
            check_arguments("synthetic_tool", &schema(), json!([1])),
            Ok(json!([1]))
        );
    }
}
